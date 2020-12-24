# -*- coding: utf-8 -*-

import codecs, torch, json, copy
from onmt.inputters.dataset_base import DatasetBase, PAD_WORD


class TextDataset(DatasetBase):
    """
    Build `Example` objects, `Field` objects, and filter_pred function
    from text corpus.

    Args:
        fields (dict): a dictionary of `torchtext.data.Field`.
            Keys are like 'src', 'tgt', 'src_map', and 'alignment'.
        src_examples_iter (dict iter): preprocessed source example
            dictionary iterator.
        tgt_examples_iter (dict iter): preprocessed target example
            dictionary iterator.
        dynamic_dict (bool)
    """
    data_type = 'text'  # get rid of this class attribute asap

    @staticmethod
    def sort_key(ex):
        if hasattr(ex, "tgt"):
            return len(ex.knl), len(ex.src), len(ex.tgt)
        return len(ex.knl), len(ex.src)

    @staticmethod
    def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs,
                             batch_dim=1, batch_offset=None):
        """
        Given scores from an expanded dictionary
        corresponeding to a batch, sums together copies,
        with a dictionary word when it is ambiguous.
        """
        offset = len(tgt_vocab)
        for b in range(scores.size(batch_dim)):
            blank = []
            fill = []
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]
            for i in range(1, len(src_vocab)):
                sw = src_vocab.itos[i]
                ti = tgt_vocab.stoi[sw]
                if ti != 0:
                    blank.append(offset + i)
                    fill.append(ti)
            if blank:
                blank = torch.Tensor(blank).type_as(batch.indices.data)
                fill = torch.Tensor(fill).type_as(batch.indices.data)
                score = scores[:, b] if batch_dim == 1 else scores[b]
                score.index_add_(1, fill, score.index_select(1, blank))
                score.index_fill_(1, blank, 1e-10)
        return scores

    @classmethod
    def make_examples(cls, sequences, truncate, side, corpus_type, model_mode):
        """
        Args:
            cls: used class
            sequences: path to corpus file or iterable
            truncate (int): maximum sequence length (0 for unlimited).
            side (str): "src" or "tgt".

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        if isinstance(sequences, str):
            sequences = cls._read_file(sequences)

        if model_mode in ['top_act', 'all_acts']:
            if side != "knl":
                # load corpus labeled DA
                if corpus_type=='train':
                    file = './data/itdd_datset_train.txt'
                if corpus_type=='valid':
                    file = './data/itdd_datset_valid.txt'
                class_label = {u'Q': 2, u'I': 3, u'C': 1, u'D': 0}
                act_labels = []
                if model_mode == 'top_act':
                    key = 'label'
                elif model_mode == 'all_acts':
                    key = 'label_all'
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        if i%4==3:
                            act_labels.append(json.loads(line)[key])


        #tmp = copy.deepcopy(sequences)
        #assert (len(act_labels) == len(list(tmp))), "Dialogue Act Dataset length is not equal to Dialoue Corpus Dataset length."  # 66332==66332

        for i, seq in enumerate(sequences):
            # the implicit assumption here is that data that does not come
            # from a file is already at least semi-tokenized, i.e. split on
            # whitespace. We cannot do modular/user-specified tokenization
            # until that is no longer the case. The fields should handle this.
            if truncate and side != "src" and side != "knl":
                seq = seq.strip().split()
                seq = seq[:truncate]
            if truncate and side == "src":
                result = []
                seq = seq.strip().split("&lt; SEP &gt;")
                for s in seq:
                    s = s.split()
                    s = s[:truncate]
                    if len(s) < truncate:
                        s += [PAD_WORD] * (truncate - len(s))
                    result += s
                seq = result
            if truncate and side == "knl":
                result = []
                seq = seq.strip().split("&lt; SEP &gt;")
                for s in seq:
                    s = s.split()
                    s = s[:truncate]
                    if len(s) < truncate:
                        s += [PAD_WORD] * (truncate - len(s))
                    result += s
                seq = result
            if not truncate and side == "tgt":
                seq = seq.strip().split()

            words, feats, _ = TextDataset.extract_text_features(seq)

            example_dict = {side: words, "indices": i}

            if feats:
                prefix = side + "_feat_"
                example_dict.update((prefix + str(j), f)
                                    for j, f in enumerate(feats))

            if model_mode in ['top_act', 'all_acts']:
                # add Dialogue Act Label to dataset(example)
                # NOTE: src-train-tokenized.txt.0.txt, tgt-train-tokenized.txt.0.txt には
                # DAラベルが入っている前提
                if side != "knl":
                    if i==0:
                        print("[onmt.inputters.text_dataset.py i==0] side: {}, model_mode: {}".format(side, model_mode))
                        print("[onmt.inputters.text_dataset.py i==0] side: {}, act_labels[i]: {}".format(side, act_labels[i]))
                if model_mode == 'top_act':
                    if side == "src":
                        example_dict.update({"src_da_label": (class_label[act_labels[i][0]], class_label[act_labels[i][1]], class_label[act_labels[i][2]])})
                    if side == "tgt":
                        example_dict.update({"tgt_da_label": (class_label[act_labels[i][3]])})
                elif model_mode == 'all_acts':
                    if side == "src":
                        example_dict.update({"src_da_label": (
                            act_labels[i][0]["I"], act_labels[i][0]["Q"], act_labels[i][0]["D"], act_labels[i][0]["C"],
                            act_labels[i][1]["I"], act_labels[i][1]["Q"], act_labels[i][1]["D"], act_labels[i][1]["C"],
                            act_labels[i][2]["I"], act_labels[i][2]["Q"], act_labels[i][2]["D"], act_labels[i][2]["C"]
                        )})
                    if side == "tgt":
                        example_dict.update({"tgt_da_label": (
                            act_labels[i][3]["I"], act_labels[i][3]["Q"], act_labels[i][3]["D"], act_labels[i][3]["C"]
                        )})

            yield example_dict

    @classmethod
    def _read_file(cls, path):
        with codecs.open(path, "r", "utf-8") as f:
            for line in f:
                yield line
