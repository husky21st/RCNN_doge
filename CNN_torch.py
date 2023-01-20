import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms = True
	torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2 ** 32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def load_CIFAR10(bs, g):
	cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
	cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

	# val_size = int(0.9 * len(cifar10_testset))
	# cifar10_valset, cifar10_testset = random_split(cifar10_testset, [val_size, len(cifar10_testset) - val_size])

	train_loader = DataLoader(cifar10_trainset, batch_size=bs, shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g)
	# val_loader = DataLoader(cifar10_valset, batch_size=int(bs/2), shuffle=False)
	test_loader = DataLoader(cifar10_testset, batch_size=int(bs/2), shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

	# return {'train': train_loader, 'val': val_loader, 'test': test_loader}
	return {'train': train_loader, 'test': test_loader}


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.conv_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
		self.conv_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
		self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
		self.linear_1 = nn.Linear(8 * 8 * 16, 128)
		self.linear_2 = nn.Linear(128, 10)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.conv_1(x))
		x = self.max_pool2d(x)
		x = self.relu(self.conv_2(x))
		x = self.max_pool2d(x)
		x = x.reshape(x.size(0), -1)
		x = self.relu(self.linear_1(x))
		x = self.linear_2(x)
		return x


if __name__ == '__main__':
	SEED = 39
	seed_everything(SEED)
	generator = torch.Generator()
	generator.manual_seed(SEED)

	# Hyper Parameters
	EPOCHS = 10
	bach_size = 128
	learning_rate = 0.001
	# image_size = 28 * 28

	model = Model()
	loaders = load_CIFAR10(bach_size, generator)

	DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
	model = model.to(DEVICE)
	print(DEVICE)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	train_loss = list()
	val_loss = list()
	val_accuracy = list()
	best_val_loss = 1
	for epoch in range(EPOCHS):
		total_train_loss = 0
		total_val_loss = 0

		model.train()
		# training
		for itr, (batch, label) in tqdm(enumerate(loaders['train'])):
			# optimizer.zero_grad()
			for param in model.parameters():
				param.grad = None

			batch = batch.to(DEVICE)
			label = label.to(DEVICE)

			outputs = model(batch)

			loss = criterion(outputs, label)
			total_train_loss += loss.item()

			loss.backward()
			optimizer.step()

		total_train_loss = total_train_loss / (itr + 1)
		train_loss.append(total_train_loss)

		# validation
		model.eval()
		correct = 0
		total = 0
		with torch.inference_mode():
			for itr, (batch, label) in tqdm(enumerate(loaders['test'])):
				batch = batch.to(DEVICE)
				label = label.to(DEVICE)

				outputs = model(batch)

				loss = criterion(outputs, label)
				total_val_loss += loss.item()

				outputs = F.softmax(outputs, dim=1)
				predicted = outputs.max(dim=1, keepdim=True)[1]
				correct += predicted.eq(label.view_as(predicted)).sum().item()
				total += label.size(0)

		accuracy = correct / total
		val_accuracy.append(accuracy)
		total_val_loss = total_val_loss / (itr + 1)
		val_loss.append(total_val_loss)

		print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, EPOCHS,
																								  total_train_loss,
																								  total_val_loss,
																								  accuracy))

		if total_val_loss < best_val_loss:
			best_val_loss = total_val_loss
			# print('Saving the net state dictionary for Epoch: {} with Validation loss: {:.8f}'.format(epoch + 1,total_val_loss))
			# torch.save(net.state_dict(), "net.dth")

	print(train_loss)
	print(val_loss)
	print(val_accuracy)
	# fig = plt.figure(figsize=(20, 10))
	# plt.plot(np.arange(1, EPOCHS + 1), train_loss, label="Train loss")
	# plt.plot(np.arange(1, EPOCHS + 1), val_loss, label="Validation loss")
	# plt.xlabel('Loss')
	# plt.ylabel('Epochs')
	# plt.title("Loss Plots")
	# plt.legend(loc='upper right')
	# plt.show()
	# # plt.savefig('loss.png')
