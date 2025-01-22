import unittest
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformer_project.schedulers.custom_scheduler import TransformerLRScheduler
from transformer_project.modelling.transformer_training import initialize_optimizer_and_scheduler

class TestTrainingComponents(unittest.TestCase):
    def test_scheduler(self):
        optimizer = torch.optim.Adam([torch.randn(10, requires_grad=True)])
        scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

        initial_lr = scheduler.get_lr()[0]
        scheduler.step()
        updated_lr = scheduler.get_lr()[0]

        self.assertNotEqual(initial_lr, updated_lr, "Learning rate should change after step()")

    def test_optimizer_initialization(self):
        model = torch.nn.Linear(10, 10)
        optimizer, scheduler = initialize_optimizer_and_scheduler(model, d_model=512, learning_rate=0.001, warmup_steps=4000)

        self.assertTrue(len(optimizer.param_groups), "Optimizer should have parameter groups.")
        self.assertIsInstance(scheduler, TransformerLRScheduler, "Scheduler should be an instance of TransformerLRScheduler.")

if __name__ == "__main__":
    unittest.main()