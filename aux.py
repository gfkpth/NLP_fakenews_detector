# auxiliary functions

# import
from nltk.stem import WordNetLemmatizer

import pandas as pd
import re
import string


# string cleaning
# currently not removing numbers, consider adding?
def cleaning_strings(iostring,lower=True):
    
    # first check for missing spaces in front of capitalised words or after punctuation
    # and insert space
    missingspace = re.compile(r'([\.]|(?:[A-Za-z]*[a-z]))([A-Z][A-Za-z]*)')
    iostring = missingspace.sub(r'\1 \2',iostring)
    
    # set up patterns for deletion
    realhyphen = re.compile(r'[^A-Za-z]+-[^A-Za-z]*')
    punctnohyphen = ''.join(list(filter(lambda x: x if x != '-' else '',string.punctuation))) + ','
    multispace = re.compile(r'\s+')
    
    # delete patterns
    wrongposs=re.compile(r'‚s')     # this removes a weirdly typeset possessive marking
    specialsigns = re.compile(f"[{re.escape(punctnohyphen)}]")
    
    # processing the patterns
    spacepattern = [realhyphen,multispace]
    delpattern = [wrongposs,specialsigns]
    for p in spacepattern:
        iostring = p.sub(' ', iostring)

    for p in delpattern:
        iostring = p.sub('', iostring)
    
    if lower:
        return iostring.lower()
    else:
        return iostring



# Your code

wordnet_lemma  = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0]#  Get POS tag's first character (e.g., 'N' from 'NN')
    #Maps it to a WordNet-compatible tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN) # returns the word type (Noun if we have not found)
    
def lemmatize(lst):
    return [wordnet_lemma.lemmatize(word,pos=get_wordnet_pos(word)) for word in lst]

    
    
    
## to be confirmed    
    


# # print evaluation and append results to results 
# def print_evaluation(model,test_ds,test_df,acc_train=None,acc_val=None,model_id=None,lr=None,epochs=None,batchsize=None,csvout=RESULT_CSV,printout=True):
#     """ Print model evaluation
    
#     Arguments:
#     - **model**: the model for evaluation
#     - **test_ds** a tensorflow dataset for the test data
#     - **test_df** a pandas dataframe for the test data (expects the true labels in 'label')
#     - **acc_train** can be used to supply the final training accuracy from the fitting history for logging (otherwise, None is supplied)
#     - **acc_val** can be used to supply the final validation accuracy from the fitting history for logging (otherwise, None is supplied)
#     - **model_id**: a string identifier for the model for logging
#     - **csvout**: filepath for csv file logging the results, set to '' to disable logging
#     - **printout**: set to False if no printed output is desired
     
#     return:
#     - classification report for storage and further comparison
#     """
#     # Predict class probabilities
#     y_pred_probs = model.predict(test_ds)

#     # Convert probabilities to class predictions
#     y_pred = y_pred_probs.argmax(axis=1)
#     y_true = test_df['label']  # original integer labels

#     # Compute metrics
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred, average='weighted')
#     rec = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')
    
#     class_names = (
#     test_df[['label', 'category']]
#     .drop_duplicates()
#     .sort_values('label')['category']
#     .tolist()
#     )

#     report = classification_report(y_true, y_pred, target_names=class_names)

#     if printout:
#         print(f"Accuracy: {acc:.4f}")
#         print(f"Precision: {prec:.4f}")
#         print(f"Recall: {rec:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(report)
    
    
#         # Predict & confusion matrix
#         cm = confusion_matrix(y_true, y_pred)

#         plt.figure(figsize=(8,6))
#         sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         plt.show()
    
#     if csvout:
#         write_results(model_id,lr,epochs,batchsize,acc_train,acc_val,acc,prec,rec,f1,file=csvout)
    
#     return report


# # function to append results to csv
# def write_results(modid,lr,epochs,batchsize,acc_train,acc_val,accuracy,precision,recall,f1,file=RESULT_CSV):

#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     fields=[timestamp, modid, lr, epochs, batchsize, acc_train,acc_val, accuracy, precision, recall, f1]
#     headers = ['time','model_id', 'learning_rate', 'epochs', 'batchsize', 'acc_train', 'acc_val', 'accuracy', 'precision', 'recall', 'f1_score']
    
    
#     file_exists = os.path.exists(file)

#     with open(file, 'a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(headers)
#         writer.writerow(fields)
        
        
# # to more easily compare
# def print_side_by_side(*tables, padding=4):
#     # Split each table into lines
#     split_tables = [t.splitlines() for t in tables]
    
#     # Find the max number of lines
#     max_lines = max(len(t) for t in split_tables)
    
#     # Pad shorter tables with empty lines
#     split_tables = [
#         t + [''] * (max_lines - len(t)) for t in split_tables
#     ]
    
#     # Join each line side-by-side
#     for lines in zip(*split_tables):
#         print((' ' * padding).join(line.ljust(30) for line in lines))