1. Design_Model_LORA.py contains the malicious design of some layers from pretrained model
2. roberta_module.py contains the code for inserting LORA modules in the architecture
3. User_Train.py contains the code for training in the user side
4. reconstruction.py contains the code for reconstructing fine-tuning sample from the gradients
5. Run Roberta_Final_Yahoo_Dataset.ipynb that will call all the functions in the above files and output recovered texts
6. This is a sample code for training on Yahoo Answers dataset
