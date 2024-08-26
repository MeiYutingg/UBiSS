import argparse, os, sys, datetime, glob

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def add_bert_parser(parser):
    parser.add_argument("--vocab_size", default=30522, type=int, required=False,
                        help="Vocabulary size")
    # multimodal transformer modeling config
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or model type.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument("--num_hidden_layers", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False,
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=-1, type=int, required=False,
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False,
                        help="Update model config if given.")
    parser.add_argument("--img_feature_dim", default=512, type=int,
                        help="Update model config if given.The Image Feature Dimension.")
    parser.add_argument("--load_partial_weights", type=str_to_bool, nargs='?',
                        const=True, default=False,
                        help="Only valid when change num_hidden_layers, img_feature_dim, but not other structures. "
                        "If set to true, will load the first few layers weight from pretrained model.")
    parser.add_argument("--freeze_embedding", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Whether to freeze word embeddings in Bert")
    parser.add_argument("--drop_out", default=0.1, type=float,
                        help="Drop out ratio in BERT.")
    # inputs to multimodal transformer config
    parser.add_argument("--max_seq_length", default=70, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_a_length", default=40, type=int,
                        help="The maximum sequence length for caption.")
    parser.add_argument("--max_img_seq_length", default=50, type=int,
                        help="The maximum total input image sequence length.")
    parser.add_argument("--do_lower_case", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--add_od_labels", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Whether to add object detection labels or not")
    parser.add_argument("--od_label_conf", default=0.0, type=float,
                        help="Confidence threshold to select od labels.")
    parser.add_argument("--use_asr", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Whether to add ASR/transcript as additional modality input")
    parser.add_argument("--unique_labels_on", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Use unique labels only.")
    parser.add_argument("--no_sort_by_conf", type=str_to_bool, nargs='?', const=True, default=False,
                        help="By default, we will sort feature/labels by confidence, "
                        "which is helpful when truncate the feature/labels.")
    #======= mask token
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help= "Probability to mask input sentence during training.")
    parser.add_argument("--max_masked_tokens", type=int, default=3,
                        help="The max number of masked tokens per sentence.")
    parser.add_argument("--attn_mask_type", type=str, default='seq2seq',
                        choices=['seq2seq', 'bidirectional', 'learn_vid_mask'], 
                        help="Attention mask type, support seq2seq, bidirectional")
    parser.add_argument("--text_mask_type", type=str, default='random',
                        choices=['random', 'pos_tag', 'bert_attn', 'attn_on_the_fly'], 
                        help="Attention mask type, support random, pos_tag, bert_attn (precomputed_bert_attn), attn_on_the_fly")
    parser.add_argument("--tag_to_mask", default=["noun", "verb"], type=str, nargs="+", 
                        choices=["noun", "verb", "adjective", "adverb", "number"],
                        help= "what tags to mask")
    parser.add_argument("--mask_tag_prob", default=0.8, type=float,
                        help= "Probability to mask input text tokens with included tags during training.")
    parser.add_argument("--tagger_model_path", type=str, default='models/flair/en-pos-ontonotes-fast-v0.5.pt', 
                        help="checkpoint path to tagger model")
    parser.add_argument("--random_mask_prob", default=0, type=float,
                        help= "Probability to mask input text tokens randomly when using other text_mask_type")
    # image feature masking (only used in captioning?)
    parser.add_argument('--mask_img_feat', type=str_to_bool,
                                nargs='?', const=True, default=False,
                                help='Enable image fetuare masking')
    parser.add_argument('--max_masked_img_tokens', type=int, default=10,
                                help="Maximum masked object featrues")

    # basic decoding configs
    parser.add_argument("--tie_weights", type=str_to_bool, nargs='?',
                                const=True, default=False,
                                help="Whether to tie decoding weights to that of encoding")
    parser.add_argument("--label_smoothing", default=0, type=float,
                                help=".")
    parser.add_argument("--drop_worst_ratio", default=0, type=float,
                                help=".")
    parser.add_argument("--drop_worst_after", default=0, type=int,
                                help=".")
    parser.add_argument('--max_gen_length', type=int, default=20,
                                help="max length of generated sentences")
    parser.add_argument('--output_hidden_states', type=str_to_bool,
                                nargs='?', const=True, default=False,
                                help="Turn on for fast decoding")
    parser.add_argument('--num_return_sequences', type=int, default=1,
                                help="repeating times per image")
    parser.add_argument('--num_beams', type=int, default=1,
                                help="beam search width")
    parser.add_argument('--num_keep_best', type=int, default=1,
                                help="number of hypotheses to keep in beam search")
    parser.add_argument('--temperature', type=float, default=1,
                                help="temperature in softmax for sampling")
    parser.add_argument('--top_k', type=int, default=0,
                                help="filter distribution for sampling")
    parser.add_argument('--top_p', type=float, default=1,
                                help="filter distribution for sampling")
    parser.add_argument('--repetition_penalty', type=int, default=1,
                                help="repetition penalty from CTRL paper "
                                "(https://arxiv.org/abs/1909.05858)")
    parser.add_argument('--length_penalty', type=int, default=1,
                                help="beam search length penalty")
    return parser