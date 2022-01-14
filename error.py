  0%|          | 0/7016 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/transformers/data/data_collator.py:317: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
(ImplicitFunc pid=3068)   sequence_length = torch.tensor(batch["input_ids"]).shape[1]
(ImplicitFunc pid=3068) /usr/local/lib/python3.7/dist-packages/transformers/data/data_collator.py:328: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
(ImplicitFunc pid=3068)   batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
(ImplicitFunc pid=3068) 2022-01-09 13:11:24,201 ERROR function_runner.py:268 -- Runner Thread raised error.
(ImplicitFunc pid=3068) Traceback (most recent call last):
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 262, in run
(ImplicitFunc pid=3068)     self._entrypoint()
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 331, in entrypoint
(ImplicitFunc pid=3068)     self._status_reporter.get_checkpoint())
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/util/tracing/tracing_helper.py", line 451, in _resume_span
(ImplicitFunc pid=3068)     return method(self, *_args, **_kwargs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 597, in _trainable_func
(ImplicitFunc pid=3068)     output = fn()
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/integrations.py", line 282, in dynamic_modules_import_trainable
(ImplicitFunc pid=3068)     return trainable(*args, **kwargs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/utils/trainable.py", line 344, in inner
(ImplicitFunc pid=3068)     trainable(config, **fn_kwargs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/integrations.py", line 183, in _objective
(ImplicitFunc pid=3068)     local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/trainer.py", line 1332, in train
(ImplicitFunc pid=3068)     tr_loss_step = self.training_step(model, inputs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/trainer.py", line 1891, in training_step
(ImplicitFunc pid=3068)     loss = self.compute_loss(model, inputs)
(ImplicitFunc pid=3068)   File "<ipython-input-32-8b083b1f0f2f>", line 29, in compute_loss
(ImplicitFunc pid=3068) RuntimeError: Could not infer dtype of NoneType
(ImplicitFunc pid=3068) Exception in thread Thread-2:
(ImplicitFunc pid=3068) Traceback (most recent call last):
(ImplicitFunc pid=3068)   File "/usr/lib/python3.7/threading.py", line 926, in _bootstrap_inner
(ImplicitFunc pid=3068)     self.run()
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 281, in run
(ImplicitFunc pid=3068)     raise e
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 262, in run
(ImplicitFunc pid=3068)     self._entrypoint()
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 331, in entrypoint
(ImplicitFunc pid=3068)     self._status_reporter.get_checkpoint())
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/util/tracing/tracing_helper.py", line 451, in _resume_span
(ImplicitFunc pid=3068)     return method(self, *_args, **_kwargs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/function_runner.py", line 597, in _trainable_func
(ImplicitFunc pid=3068)     output = fn()
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/integrations.py", line 282, in dynamic_modules_import_trainable
(ImplicitFunc pid=3068)     return trainable(*args, **kwargs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/ray/tune/utils/trainable.py", line 344, in inner
(ImplicitFunc pid=3068)     trainable(config, **fn_kwargs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/integrations.py", line 183, in _objective
(ImplicitFunc pid=3068)     local_trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/trainer.py", line 1332, in train
(ImplicitFunc pid=3068)     tr_loss_step = self.training_step(model, inputs)
(ImplicitFunc pid=3068)   File "/usr/local/lib/python3.7/dist-packages/transformers/trainer.py", line 1891, in training_step
(ImplicitFunc pid=3068)     loss = self.compute_loss(model, inputs)
(ImplicitFunc pid=3068)   File "<ipython-input-32-8b083b1f0f2f>", line 29, in compute_loss
(ImplicitFunc pid=3068) RuntimeError: Could not infer dtype of NoneType
(ImplicitFunc pid=3068) 