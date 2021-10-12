import torch


class TrainLogger:
    def __init__(self, debug_step: int):
        self._debug_step = debug_step
        self._running_loss = 0.0
        self._running_regression_loss = 0.0
        self._running_classification_loss = 0.0

    def apply_log_step(self, loss, regression_loss, classification_loss, current_step):
        self._running_loss += loss.item()
        self._running_regression_loss += regression_loss.item()
        self._running_classification_loss += classification_loss.item()

        if current_step != 0 and current_step % self._debug_step == 0:
            avg_loss = self._running_loss / self._debug_step
            avg_reg_loss = self._running_regression_loss / self._debug_step
            avg_clf_loss = self._running_classification_loss / self._debug_step

            print(
                f'Step: {current_step}\n' +
                f'Average Loss: {avg_loss:.4f}\n' +
                f'Average Regression Loss: {avg_reg_loss:.4f}\n' +
                f'Average Classification Loss: {avg_clf_loss:.4f}\n'
            )
            self._running_loss = 0.0
            self._running_regression_loss = 0.0
            self._running_classification_loss = 0.0


class Fitter:
    def __init__(self,
                 net,
                 loss_function,
                 optimizer,
                 device,
                 log_training=False,
                 debug_step=5,
                 ):
        self._net = net
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._device = device
        self._log_training = log_training
        if log_training:
            self._logger = TrainLogger(debug_step=debug_step)

    def train(self, data_loader):
        self._net.train(True)

        for i, data in enumerate(data_loader):
            images, boxes, class_ids = data
            images = images.to(self._device)
            boxes = boxes.to(self._device)
            class_ids = class_ids.to(self._device)

            self._optimizer.zero_grad()
            confidences, locations = self._net(images)

            regression_loss, classification_loss = self._loss_function(confidences, locations, class_ids, boxes)
            loss = regression_loss + classification_loss
            loss.backward()
            self._optimizer.step()
            if self._log_training and self._logger:
                self._logger.apply_log_step(loss, regression_loss, classification_loss, current_step=i)

    def validate(self, data_loader):
        self._net.eval()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        samples_count = 0
        for _, data in enumerate(data_loader):
            images, boxes, class_ids = data
            images = images.to(self._device)
            boxes = boxes.to(self._device)
            class_ids = class_ids.to(self._device)
            samples_count += 1

            with torch.no_grad():
                confidence, locations = self._net(images)
                regression_loss, classification_loss = self._loss_function(confidence, locations, class_ids, boxes)
                loss = regression_loss + classification_loss

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()

        running_loss = running_loss / samples_count
        running_regression_loss = running_regression_loss / samples_count
        running_classification_loss = running_classification_loss / samples_count

        return running_loss, running_regression_loss, running_classification_loss
