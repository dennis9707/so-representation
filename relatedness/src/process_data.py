import pandas as pd
import codecs
import argparse
from tqdm import tqdm
def decoder(value):
    from codecs import encode, decode
    a = decode(encode(value, 'latin-1', 'backslashreplace'), 'unicode-escape')
    b = decode(encode(a, 'latin-1', 'backslashreplace'), 'unicode-escape')
    c = b.replace("\\"," ")
    return c

def process_csv(file_path):
    df = pd.read_csv(file_path, sep=",", header=0, lineterminator='\n')
    for index, row in df.iterrows():
        if row['q1_Body'] == '\\N':
            df.set_value(index,'q1_Body',"")

        if row['q1_BodyCode'] == '\\N':
            df.set_value(index,'q1_BodyCode',"")

        if row['q1_AcceptedAnswerId'] == '\\N':
            df.set_value(index,'q1_AcceptedAnswerId',"")
        
        if row['q1_AcceptedAnswerBody'] == '\\N':
            df.set_value(index,'q1_AcceptedAnswerBody',"")
        
        if row['q1_AcceptedAnswerCode'] == '\\N':
            df.set_value(index,'q1_AcceptedAnswerCode',"")
        
        if row['q1_AnswersIdList'] == '\\N':
            df.set_value(index,'q1_AnswersIdList',"")

        if row['q1_AnswersBody'] == '\\N':
            df.set_value(index,'q1_AnswersBody',"")

        if row['q1_AnswersCode'] == '\\N':
            df.set_value(index,'q1_AnswersCode',"")
            
        if row['q2_Body'] == '\\N':
            df.set_value(index,'q2_Body',"")

        if row['q2_BodyCode'] == '\\N':
            df.set_value(index,'q2_BodyCode',"")

        if row['q2_AcceptedAnswerId'] == '\\N':
            df.set_value(index,'q2_AcceptedAnswerId',"")
        
        if row['q2_AcceptedAnswerBody'] == '\\N':
            df.set_value(index,'q2_AcceptedAnswerBody',"")
        
        if row['q2_AcceptedAnswerCode'] == '\\N':
            df.set_value(index,'q2_AcceptedAnswerCode',"")
        
        if row['q2_AnswersIdList'] == '\\N':
            df.set_value(index,'q2_AnswersIdList',"")

        if row['q2_AnswersBody'] == '\\N':
            df.set_value(index,'q2_AnswersBody',"")
            
        if row['q2_AnswersCode'] == '\\N':
            df.set_value(index,'q2_AnswersCode',"")

    cnt = 0
    ignore = [0,1,12]
    for rowIndex, row in df.iterrows(): #iterate over rows
        for columnIndex, value in row.items():
            if isinstance(value, str):
                decode_value = decoder(value)
                df.at[rowIndex,columnIndex] = decode_value
                cnt += 1
    return df
    

def conc_csv(df, output_path):
    df = df.fillna('')
    all_sent1 = []
    all_sent2 = []
    all_label = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):  
        post1, post2, label = '', '', ''
        post1 += row['q1_Title'] + \
                row['q1_Body'] + \
                row['q1_BodyCode'] + \
                row['q1_Tags'] + \
                row['q1_AcceptedAnswerBody'] + \
                row['q1_AnswersBody'] + \
                row['q1_AcceptedAnswerCode'] + \
                row['q1_AnswersCode']

        post2 += row['q2_Title'] + \
                row['q2_BodyCode'] + \
                row['q2_Body'] + \
                row['q2_Tags'] + \
                row['q2_AcceptedAnswerBody'] + \
                row['q1_AnswersBody'] + \
                row['q2_AcceptedAnswerCode'] + \
                row['q1_AnswersCode']
  
        label = row['class']
        post1 = ' '.join(post1.split())
        post2 = ' '.join(post2.split())
        assert( post1!='' and post2!='' and label!='')
        all_sent1.append(post1)
        all_sent2.append(post2)
        all_label.append(label.strip())
    assert(len(all_sent1) == len(all_sent2) == len(all_label))
            

    print(len(all_sent1), len(all_label))

    data_dict = { 'sentence1': all_sent1 ,'sentence2': all_sent2, 'label': all_label}
    df = pd.DataFrame(data_dict)
    df.to_csv(output_path,index=False)

def main():
    parser = argparse.ArgumentParser(description='preprocess csv file for relatedness task')
    parser.add_argument('--input', type=str, default="../data/train.csv", help='input directory path which stores splited csv')
    parser.add_argument('--output', type=str, default="../data/train_process.csv", help='output directory path which stores processed csv')

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    df = process_csv(input_path)
    conc_csv(df, output_path)

if __name__ == "__main__":
    main()