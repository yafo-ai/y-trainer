import jsonlines

class TokenLossRecord:
    def __init__(self, data_id, token_id, global_step, step, loss, weighted_loss, sample_id, position, epoch):
        self.data_id = data_id
        self.token_id = token_id
        self.global_step = global_step
        self.step = step
        self.loss = loss
        self.weighted_loss = weighted_loss
        self.sample_id = sample_id
        self.position = position
        self.epoch = epoch
        
    def to_dict(self):
        return {
            "data_id": self.data_id,
            "token_id": self.token_id,
            "global_step": self.global_step,
            "loss": self.loss,
            "weighted_loss": self.weighted_loss,
            "sample_id": self.sample_id,
            "position": self.position,
            "step": self.step,
            "epoch": self.epoch
        }

def save_token_records(records, file_path):
    """"""
    with jsonlines.open(file_path, 'w') as f:
        for record in records:
            f.write(record.to_dict())
