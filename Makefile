export CUDA_VISIBLE_DEVICES=0

train:
	python main.py -mode train -model_name $(name) -language ${language} -train_path dataset/$(dataset)/train.csv -num_classes 3

test:
	python main.py -mode test -model_name $(name) -language ${language} -test_path dataset/$(dataset)/test.csv -num_classes 3
