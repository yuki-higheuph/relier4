import os

import pandas as pd
import torch
from flask import Flask, render_template, request, redirect
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# BERTモデルの読み込み
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # ポジティブ／ネガティブの2つのラベル

# ファインチューニングの設定
BATCH_SIZE = 8
MAX_LENGTH = 32
LEARNING_RATE = 2e-5
EPOCHS = 10

# テキストのポジティブ／ネガティブ判定
def analyze_sentiment_with_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    positive_probability = probabilities[:, 1].item()  # ポジティブである確率
    if positive_probability > 0.5:
        return 'Positive'
    else:
        return 'Negative'

# ポジティブなコメントにはポイントを加算し、ネガティブなコメントにはポイントを減算
def calculate_points_based_on_sentiment(sentiment):
    if sentiment == 'Positive':
        return 5  # ポジティブなコメントには5ポイント加算
    elif sentiment == 'Negative':
        return -5  # ネガティブなコメントには5ポイント減算
    else:
        return 0

# メッセージ表示
def display_message(points):
    if points <= 30:
        return "対象者はあなたのコミュニティの中で危険な言葉を多く使われている様子なので、注意を払った方がよいかもしれません"
    elif points <= 70:
        return "対象者はあなたのコミュニティの中で特に問題となる発言は見受けられませんでした"
    else:
        return "対象者はあなたのコミュニティの中で問題となる発言は見受けられず、手本となる発言が多いようです"

# データセットクラスを定義
class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'comment']
        label = self.data.loc[idx, 'sentiment']

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        inputs = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(1 if label == 'Positive' else 0)  # ポジティブなら1、ネガティブなら0
        }
        return inputs

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # CSVファイルを読み込む
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print("utf-8 で読み込み成功")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding='shift_jis')
                print("shift_jis で読み込み成功")
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='cp932')
                print("cp932 で読み込み成功")

        # 'comment' 列を文字列に変換
        df['comment'] = df['comment'].astype(str)

        # データセットを作成
        dataset = SentimentDataset(df, tokenizer, max_length=MAX_LENGTH)
        # データローダーを作成
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # オプティマイザーとロス関数を定義
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss()

        # トレーニングループ
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            average_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {average_loss:.4f}')

        # データフレームの内容をHTMLに渡すために変換
        data_html = df.to_html()

        results = []
        for _ in range(100):
            df['感情分析'] = df['comment'].apply(analyze_sentiment_with_bert)
            # ポイント計算
            df['ポイント'] = df['感情分析'].apply(calculate_points_based_on_sentiment)
            final_points = 50 + df['ポイント'].sum()
            results.append(final_points)

        # 平均値を計算し、小数点第3位を切り捨てて小数点第2位まで表示
        average_points = round(sum(results) / len(results), 2)

        # メッセージ表示
        final_message = display_message(average_points)

        return render_template('index.html', data_html=data_html, average_points=average_points, final_message=final_message )
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)