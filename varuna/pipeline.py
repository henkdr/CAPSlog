
import torch
from torch import Tensor, nn
import torch.distributed as dist

from torchviz import make_dot, make_dot_from_trace

from queue import Queue
from threading import Thread
try:
    from apex import amp
    from apex.amp import _amp_state
    import amp_C, apex_C
except:
    pass

import math
import os, sys
import time

class Pipeline:
    """ Pipeline parallelism for Varuna """

    def __init__(self, batches, model, config, schedule, optimizer, verbose=False):
        self.batches = batches
        self.chunks = len(self.batches)

        self.model = model
        self.partitioned_model = self.model
        self.schedule = schedule
        self.rank = dist.get_rank()
        self.opportunistic = True
        self.verbose = verbose

        self.read_config(config)

        if self.make_logfile:
            replica_num = self.stage_to_rank_map[self.stage].index(self.rank)
            microBS = config["chunk_size"]
            logfilename = "varuna_logs-"+str(self.data_depth)+"dp-" + str(microBS) + "mBS-stage" + str(self.stage) + "of" + str(self.partitions) + "_" + str(replica_num)
            # logfilename = os.path.join("/home/varuna/gpt2-blob/perf_analysis_2.5b","stats",logfilename)
            self.logfile = open(logfilename,"a")
            self.logfile.write("start time {}\n".format(time.time()))

        self.optimizer = optimizer

        self.spawn_send_workers()
        # self.spawn_receive_workers()
        self.acts_queue = Queue()
        self.grads_queue = Queue()
        self.recompute_queue = Queue()
        self.excp_queue = Queue()

        # self.back_start_times = Queue()

        # communication queues
        self.partitioned_model.set_queues(self.acts_send_queue, self.grads_send_queue,
                                          self.acts_queue, self.grads_queue, self.recompute_queue, self.excp_queue)

        # stores output of recompute(/forward) pass to be used by backward()
        self.loss = None
        self.average_loss = 0

        self.pre_fwd_events = []
        self.post_fwd_events = []
        self.avg_fwd_time = 0

    def read_config(self, config):

        self.partitions = config["partitions"]
        self.stage = config["stage"]
        self.pipeline_group = config["pipeline_process_group"]
        self.rank_within_stage = config["rank_within_stage"]

        self.device = config["device"]
        self.fp16 = config["fp16"]

        self.fwd_inp_shape = config["fwd_inp_shape"]
        self.fwd_inp_shape_changes = config["fwd_inp_shape_changes"]
        self.bwd_grad_shape = config["bwd_grad_shape"]
        self.bwd_grad_shape_changes = config["bwd_grad_shape_changes"]
        self.parameter_names = config["parameter_names"]

        self.stage_to_rank_map = config["stage_to_rank_map"]
        self.local_rank = config["local_rank"]

        self.make_logfile = config["make_logfile"]
        self.receive_rank = config["receive_rank"]
        self.send_rank = config["send_rank"]
        self.last_chunk_size = config["last_chunk_size"]
    
    def spawn_receive_workers(self):
        self.acts_receive_thread = None
        self.grads_receive_thread = None

        if self.stage > 0:
            self.acts_receive_thread = Thread(target=self.acts_receiver, args=())
            self.acts_receive_thread.daemon=True
            self.acts_receive_thread.start()

        if self.stage < self.partitions-1:
            self.grads_receive_thread = Thread(target=self.grads_receiver, args=())
            self.grads_receive_thread.daemon=True
            self.grads_receive_thread.start()
    
    def spawn_send_workers(self):
        self.grads_send_queue = Queue()
        self.acts_send_queue = Queue()
        self.acts_send_thread = None
        self.grads_send_thread = None

        if self.stage < self.partitions-1:
            self.acts_send_thread = Thread(target=self.acts_sender, args=())
            self.acts_send_thread.daemon=True
            self.acts_send_thread.start()

        if self.stage > 0:
            self.grads_send_thread = Thread(target=self.grads_sender, args=())
            self.grads_send_thread.daemon=True
            self.grads_send_thread.start() 
    
    def acts_receiver(self):
        chunks = len(self.batches)
        dtype = torch.float16 if self.fp16 else torch.float32
        recv_handles = Queue()

        for task,index in self.schedule:
            if task == 0:
                try:
                    tensors = [None] * len(self.fwd_inp_shape)

                    for i, fwd_inp_shape in enumerate(self.fwd_inp_shape):
                        if index == (chunks-1) and self.last_chunk_size > 0:
                            for d in self.fwd_inp_shape_changes[i]:
                                fwd_inp_shape[d] = self.last_chunk_size

                        tag_id = i + (index *  len(self.fwd_inp_shape))

                        tensors[i] = torch.ones(fwd_inp_shape, dtype=dtype)
                        handle = dist.irecv(tensors[i], src=self.receive_rank, tag=tag_id)
                        recv_handles.put(handle)

                    # if recv_handles.qsize()>4:
                    #     handle, tensor = recv_handles.get()
                    #     handle.wait()
                    #     self.acts_queue.put(tensor)
                    while not recv_handles.empty():
                        handle = recv_handles.get()
                        handle.wait()
                    self.acts_queue.put(tensors)
                except Exception as e:
                    self.excp_queue.put(e)
                    return
        del tensors
    
    def grads_receiver(self):
        chunks = len(self.batches)
        tensors_per_chunk = len(self.bwd_grad_shape)
        dtype = torch.float16 if self.fp16 else torch.float32
        recv_handles = Queue()

        for task,index in self.schedule:
            if task == 2:
                try:
                    tensors = [None] * tensors_per_chunk

                    for i, bwd_grad_shape in enumerate(self.bwd_grad_shape):
                        if index == (chunks-1) and self.last_chunk_size > 0:
                            for d in self.bwd_grad_shape_changes[i]:
                                bwd_grad_shape[d] = self.last_chunk_size

                        tensors[i] = torch.ones(bwd_grad_shape, dtype=dtype)
                        # tag unique to this tensor in this micro-batch
                        # gradient tags are negative
                        tag_id = (chunks * tensors_per_chunk) + (i + (index * tensors_per_chunk))
                        handle = dist.irecv(tensors[i], src=self.send_rank, tag=tag_id)
                        recv_handles.put(handle)

                        # if recv_handles.qsize()>4:
                        #     handle, tensor = recv_handles.get()
                        #     handle.wait()
                        #     self.grads_queue.put(tensor)
                    while not recv_handles.empty():
                        handle = recv_handles.get()
                        handle.wait()
                    self.grads_queue.put(tensors)
                except Exception as e:
                    # in case connection is closed
                    self.excp_queue.put(e)
                    return
        del tensors

    def acts_sender(self):
        count = 0
        for task,index in self.schedule:
            if task == 0:
                count += 1
        
        send_handles = Queue()
        indexing_count = count
        while count > 0:
            output_acts = self.acts_send_queue.get() # list of acts
            for i, act in enumerate(output_acts):
                tag_id = i + ((indexing_count - count) *  len(self.bwd_grad_shape))
                handle = dist.isend(act.contiguous(), dst=self.send_rank, tag=tag_id)
                send_handles.put(handle)
            if send_handles.qsize() > len(output_acts):
                handle = send_handles.get()
                handle.wait()
            count -= 1
        while not send_handles.empty():
            handle = send_handles.get()
            handle.wait()

    def grads_sender(self):
        chunks = len(self.batches)
        tensors_per_chunk = len(self.fwd_inp_shape)

        count = 0
        for task,index in self.schedule:
            if task == 2:
                count += 1
        
        send_handles = Queue()
        indexing_count = count
        while count > 0:
            input_grads = self.grads_send_queue.get()
            for i, grad in enumerate(input_grads):
                tag_id = (chunks * tensors_per_chunk) + (i + ((indexing_count - count) * tensors_per_chunk))
                handle = dist.isend(grad.contiguous(), dst=self.receive_rank, tag=tag_id)
                send_handles.put(handle)
            if send_handles.qsize()>len(input_grads):
                handle = send_handles.get()
                handle.wait()
            count -= 1
        while not send_handles.empty():
            handle = send_handles.get()
            handle.wait()
        
    def close_comm_threads(self):
        if self.acts_receive_thread is not None:
            self.acts_receive_thread.join()
        if self.grads_receive_thread is not None:
            self.grads_receive_thread.join()
        
        if self.acts_send_thread is not None:
            self.acts_send_thread.join()
        if self.grads_send_thread is not None:
            self.grads_send_thread.join()

    def worker(self, task, grad_mode, inputs_as_dict):
        """ Main body of worker loop """
        # forward
        if task == 0:
            torch.set_grad_enabled(grad_mode)
            if torch.cuda.is_available():
                pre_fwd = torch.cuda.Event(enable_timing=True)
                post_fwd = torch.cuda.Event(enable_timing=True)
                pre_fwd.record()
            output = self.model(inputs_as_dict, save_ctx=not grad_mode, handle_comm=True)

            # compgraph = make_dot(output, params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True)
            # filename = "compgraph_gpu{}".format(self.rank)
            # compgraph.filename = filename
            # compgraph.render()

            if torch.cuda.is_available():
                post_fwd.record()
                self.pre_fwd_events.append(pre_fwd)
                self.post_fwd_events.append(post_fwd)

            if grad_mode == True:
                # save loss and input activations for the backward pass to use
                self.loss = output[0] if isinstance(output,tuple) else output

        # recompute
        elif task == 1:
            torch.set_grad_enabled(True)
            output = self.model(inputs_as_dict, recompute=True, handle_comm=True)

            # compgraph = make_dot(output, params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True)
            # filename = "compgraph_recompute_gpu{}".format(self.rank)
            # compgraph.filename = filename
            # compgraph.render()

            self.loss = output[0] if isinstance(output,tuple) else output
        
        # backward
        else:
            grads = torch.ones(self.loss.size(), dtype = torch.float32).to(self.device)

            if self.stage == self.partitions - 1:
                grads = None
                self.loss = self.loss/self.chunks
                self.average_loss += (self.loss.item())

            if self.fp16:
                with amp.scale_loss(self.loss, self.optimizer, delay_overflow_check=True, 
                            last_partition=(self.stage == self.partitions-1)) as scaled_loss:
                    scaled_loss.backward(grads)
            else:
                self.loss.backward(grads)

            del self.loss
            self.loss = None
        
    def run(self):
        if self.verbose:
            print(f'{self.rank} {self.rank_within_stage} starting pipeline')        

        self.spawn_receive_workers()
        batchstart = time.time()

        schedule = [s for s in enumerate(self.schedule)]
        i=0
        count_fwd = 0
        while (i<len(schedule)):
            grad_mode = False
            index, task = schedule[i]
            # dynamic schedule - run forward if gradients for backward are not ready yet
            if self.opportunistic and (task[0]==1 and count_fwd<len(self.batches) and self.grads_queue.empty()):
            # if (task[0]==1 and count_fwd<len(self.batches) and not self.acts_queue.empty()):
                j=i
                while (j<len(schedule)):
                    if (schedule[j][1][0]==0):
                        index, task = schedule[j]
                        schedule.insert(i, schedule[j])
                        del schedule[j+1]
                        break
                    j+=1
            if (task[0]==0):
                count_fwd+=1
                if (self.schedule[index+1][0]==2):      # if next task in schedule is backward  -- no recomputation
                    grad_mode=True
            
            if self.verbose:
                print(f'{self.stage} {self.rank_within_stage} task:{task[0]} {task[1]}/{len(self.batches)}\n', end="")

            try:
                self.worker(task[0], grad_mode, self.batches[task[1]])
            except Exception as e:
                dist.destroy_process_group()
                sys.exit("Error occurred, exiting!")

            i+=1
        
        if self.device != "cpu":
            torch.cuda.synchronize(self.device)
            if len(self.pre_fwd_events) > 0:
                avg_fwd_time = 0.0
                for i in range(len(self.pre_fwd_events)):
                    start = self.pre_fwd_events[i]
                    end = self.post_fwd_events[i]
                    avg_fwd_time += start.elapsed_time(end)
                avg_fwd_time = avg_fwd_time / len(self.pre_fwd_events)
                self.avg_fwd_time = avg_fwd_time

        if self.pipeline_group is not None:
            torch.distributed.barrier(group=self.pipeline_group)
        self.close_comm_threads()
        dist.barrier()
        return self.average_loss, self.avg_fwd_time
