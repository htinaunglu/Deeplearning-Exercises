from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

hyperparameters = {"epochs": "2", "batch-size": "32", "test-batch-size": "100", "lr": "0.001"}

estimator = PyTorch(
    entry_point="scripts/pytorch_cifar.py",
    base_job_name="sagemaker-script-mode",
    role=get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.large",
    hyperparameters=hyperparameters,
    framework_version="1.8",
    py_version="py36",
)

estimator.fit(wait=True)