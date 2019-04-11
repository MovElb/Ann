import random


class BatchGen:
    """Batch generator class which can be used either for training or inference.

    """

    def __init__(self, data, batch_size, evaluation=False):
        """
        Args:
            data (list of lists): raw preprocessed data
            batch_size (int): number of elements in batch
            evaluation (bool): flag indicating current mode

        """
        self.batch_size = batch_size

        # shuffle
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        """
        Yields:
             batch (list of lists): len(batch) <= batch_size (may be less for last batch). Contains following data:
                [0] - context_ids
                [1] - context_tokens
                [2] - context_features
                [3] - context_tag_ids
                [4] - context_ent_ids
                [5] - question_ids
                [6] - question_tokens
                [7] - question_features
                [8] - question_tag_ids
                [9] - question_ent_ids
                [10] - context_token_spans
                [11] - raw context
                [12] - raw question
            when not evaluation:
                [13] - has_answer
                [14] - answer_start
                [15] - answer_end
                [16] - plausible_answer_start
                [17] - plausible_answer_end
        Note:
            tag - Part of speech tag
            ent - Named entity
        """
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data[i:i + self.batch_size]
            batch = list(zip(*batch_data))
            yield batch
