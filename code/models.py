import torch
from torch import nn
from copy import deepcopy


class VETNet(nn.Module):
    def __init__(
        self, timeseries_size, scanpath_size, cnn_shape=(16, 6), vector_shape=(256, 50)
    ):
        super(VETNet, self).__init__()
        self.scanpath_layer = nn.Sequential(
            nn.Conv2d(scanpath_size[0], cnn_shape[0], kernel_size=5, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_shape[0], cnn_shape[1], kernel_size=5, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(
                scanpath_size[1] * scanpath_size[2] * cnn_shape[1] // 16,
                vector_shape[1],
            ),
        )

        self.timeseries_layer_attention = nn.MultiheadAttention(
            timeseries_size[-1], 1, batch_first=True
        )
        self.timeseries_layer_gru = nn.GRU(
            input_size=timeseries_size[-1],
            hidden_size=vector_shape[0],
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(vector_shape[0] + vector_shape[1], 20),
            nn.Linear(20, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x_timeseries, x_visual):
        x_visual = self.scanpath_layer(x_visual / 128 - 1)

        x_timeseries, _ = self.timeseries_layer_attention(
            x_timeseries, x_timeseries, x_timeseries
        )
        _, x_timeseries = self.timeseries_layer_gru(x_timeseries)

        x_concat = torch.cat((torch.squeeze(x_timeseries, 0), x_visual), dim=1)

        return self.classifier(x_concat)

    def train(
        self,
        optimizer,
        criterion,
        train_loader,
        epochs,
        val_criterion=None,
        val_loader=[],
        lr_tracker=None,
        earlystop_tracker=None,
        device=None,
    ):

        graph = []
        running_loss = []
        val_running_loss = []

        for epoch in range(0, epochs):  # loop over the dataset multiple times

            running_loss += [0.0]
            val_running_loss += [0.0]

            for inputs, labels in train_loader:
                if isinstance(inputs, list):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].to(device)
                else:
                    inputs = [inputs.to(device)]
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(*inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss[-1] += loss.item()

            for val_inputs, val_labels in val_loader:
                if isinstance(val_inputs, list):
                    for i in range(len(inputs)):
                        val_inputs[i] = val_inputs[i].to(device)
                else:
                    val_inputs = [val_inputs.to(device)]

                val_labels = val_labels.to(device)

                val_outputs = self(*val_inputs)
                val_loss = val_criterion(val_outputs, val_labels)
                val_running_loss[-1] += val_loss.item()

            graph += [
                [
                    epoch,
                    optimizer.param_groups[-1]["lr"],
                    val_running_loss[-1],
                    running_loss[-1],
                ]
            ]

            if lr_tracker is not None:
                lr_tracker.check(
                    value=running_loss[-1], optimizer=optimizer, model=self
                )

            if earlystop_tracker is not None:
                if earlystop_tracker.check(value=running_loss[-1], model=self):
                    break
        
        print(epoch, running_loss[-1], optimizer.param_groups[-1]["lr"])
        return graph
    
    # define init method inside your model class
    def init_with_normal(self):
        def weights_init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
            elif isinstance(m, nn.GRU) or isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.normal_(param.data)

        self.apply(weights_init)


class VETNet_score(nn.Module):
    def __init__(
        self, timeseries_size, scanpath_size, cnn_shape=(16, 6), vector_shape=(256, 50)
    ):
        super(VETNet_score, self).__init__()
        self.scanpath_layer = nn.Sequential(
            nn.Conv2d(scanpath_size[0], cnn_shape[0], kernel_size=5, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_shape[0], cnn_shape[1], kernel_size=5, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(
                scanpath_size[1] * scanpath_size[2] * cnn_shape[1] // 16,
                vector_shape[1],
            ),
        )

        self.timeseries_layer_attention = nn.MultiheadAttention(
            timeseries_size[-1] - 1, 1, batch_first=True
        )
        self.timeseries_layer_gru = nn.GRU(
            input_size=timeseries_size[-1] - 1,
            hidden_size=vector_shape[0],
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(vector_shape[0] + vector_shape[1] + 1, 20),
            nn.Linear(20, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x_timeseries, x_scanpath, x_score):
        x_scanpath = self.scanpath_layer(x_scanpath)

        x_timeseries, _ = self.timeseries_layer_attention(
            x_timeseries, x_timeseries, x_timeseries
        )
        _, x_timeseries = self.timeseries_layer_gru(x_timeseries)

        x = torch.cat(
            (
                torch.squeeze(x_timeseries, 0),
                x_scanpath,
                torch.unsqueeze(x_score, dim=-1),
            ),
            dim=1,
        )

        return self.classifier(x)


class Subject_classifier(nn.Module):
    def __init__(self):
        super(Subject_classifier, self).__init__()

        self.classifier = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=1))

    def forward(self, x):
        return self.classifier(x)


class CNN(nn.Module):
    def __init__(self, scanpath_size, cnn_shape=(16, 6), vector_shape=50):
        super(CNN, self).__init__()
        self.scanpath_layer = nn.Sequential(
            nn.Conv2d(scanpath_size[0], cnn_shape[0], kernel_size=5, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_shape[0], cnn_shape[1], kernel_size=5, padding="same"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(
                scanpath_size[1] * scanpath_size[2] * cnn_shape[1] // 16, vector_shape
            ),
        )

        self.classifier = nn.Sequential(
            nn.Linear(vector_shape, 20), nn.Linear(20, 2), nn.Softmax(dim=1)
        )

    def forward(self, x_scanpath):
        x_scanpath = self.scanpath_layer(x_scanpath)

        return self.classifier(x_scanpath)


class GRUNet(nn.Module):
    def __init__(self, timeseries_size, vector_shape=50):
        super(GRUNet, self).__init__()

        self.timeseries_layer_attention = nn.MultiheadAttention(
            timeseries_size[-1] - 1, 1, batch_first=True
        )
        self.timeseries_layer_gru = nn.GRU(
            input_size=timeseries_size[-1] - 1,
            hidden_size=vector_shape,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(vector_shape, 20), nn.Linear(20, 2), nn.Softmax(dim=1)
        )

    def forward(self, x_timeseries):

        x_timeseries, _ = self.timeseries_layer_attention(
            x_timeseries, x_timeseries, x_timeseries
        )
        _, x_timeseries = self.timeseries_layer_gru(x_timeseries)

        x = torch.squeeze(x_timeseries, 0)
        return self.classifier(x)


class EarlyStopping:
    def __init__(self, patience: int, mode: str, minimum_delta=0.0):
        assert mode in {"min", "max"}, "mode has to be 'min' or 'max'"
        self.minimum_delta = minimum_delta
        self.patience = patience
        self.counter = 0
        self.tracking = torch.inf if mode == "min" else -torch.inf
        self.mode = 1 if mode == "max" else -1
        self.best_model = None

    def check(self, value, model: nn.Module):
        if self.mode * (value - self.tracking) <= self.minimum_delta:
            self.counter += 1
        else:
            self.counter = 0
            self.tracking = value
            self.best_model = deepcopy(model.state_dict())

        if self.counter == self.patience:
            model.load_state_dict(self.best_model)
            return True

        return False


class ReduceLROnPlateau:
    def __init__(
        self, patience: int, rate: float, mode: str, minimum_lr=0.0, minimum_delta=0.0
    ):
        assert rate <= 1 and rate > 0, "rate as to be a number between 0 and 1"
        assert mode in {"min", "max"}, "mode has to be 'min' or 'max'"

        self.minimum_delta = minimum_delta
        self.patience = patience
        self.counter = 0
        self.rate = rate
        self.minimum_lr = minimum_lr
        self.tracking = torch.inf if mode == "min" else -torch.inf
        self.mode = 1 if mode == "max" else -1
        self.best_model = None

    def check(self, value, optimizer, model):
        if self.mode * (value - self.tracking) <= self.minimum_delta:
            self.counter += 1
        else:
            self.counter = 0
            self.tracking = value
            self.best_model = deepcopy(model.state_dict())

        if self.counter == self.patience:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = max(
                    self.rate * optimizer.param_groups[i]["lr"], self.minimum_lr
                )
            model.load_state_dict(self.best_model)
            self.counter = 0
