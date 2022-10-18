import torch
from summarization.models.model_builder import ExtSummarizer
from summarization.ext_sum import summarize

def sum(input_fp, result_fp, model_type):
	# Load model
	# model_type = 'distilbert' #@param ['bertbase', 'distilbert', 'mobilebert']
	checkpoint = torch.load(f'summarization/checkpoints/{model_type}_ext.pt', map_location = 'cpu')
	model = ExtSummarizer(checkpoint = checkpoint, bert_type = model_type, device = 'cpu')

	# Run summarization
	# input_fp = 'raw_data/input2.txt'
	summary = summarize(input_fp, result_fp, model, max_length = 10)
