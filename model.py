import torch
import torch.nn as nn

model_task_1 = nn.Sequential(
    nn.Flatten(),  # Преобразуем (1, 28, 28) в (784,)
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Проверка размерностей
x = torch.randn(1, 28, 28)  # Тестовый вход
print(f"Input shape: {x.shape}")
output = model_task_1(x)
print(f"Output shape: {output.shape}") 