import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

class VideoCommentAnalyzer:
    """视频评论分析器"""
    
    def __init__(self, top_n_words=10):
        self.top_n_words = top_n_words
        self.video_data = None
        self.comments_data = None
        self.predictors = {}
        
    def load_data(self, video_csv_path="origin_videos_data.csv", 
                  comments_csv_path="origin_comments_data.csv"):
        """加载数据文件"""
        print("正在加载数据...")
        self.video_data = pd.read_csv(video_csv_path)
        self.comments_data = pd.read_csv(comments_csv_path)
        print(f"视频数据: {len(self.video_data)} 条")
        print(f"评论数据: {len(self.comments_data)} 条")
        
    def preprocess_video_data(self):
        """预处理视频数据"""
        self.video_data["text"] = (
            self.video_data["video_desc"].fillna("") + " " + 
            self.video_data["video_tags"].fillna("")
        )
        
    def predict_product_names(self):
        """预测产品名称"""
        print("正在预测产品名称...")

        has_product_name = ~self.video_data["product_name"].isnull()
        
        if has_product_name.sum() == 0:
            print("警告: 没有产品名称标签数据，跳过预测")
            return

        self.predictors['product_name'] = make_pipeline(
            TfidfVectorizer(tokenizer=jieba.lcut, max_features=50),
            SGDClassifier(random_state=42)
        )

        self.predictors['product_name'].fit(
            self.video_data[has_product_name]["text"],
            self.video_data[has_product_name]["product_name"]
        )
   
        self.video_data["product_name"] = self.predictors['product_name'].predict(
            self.video_data["text"]
        )
        
    def predict_comment_categories(self):
        """预测评论分类"""
        print("正在预测评论分类...")

        columns_to_predict = ['sentiment_category', 'user_scenario', 
                             'user_question', 'user_suggestion']
        
        for col in columns_to_predict:
            if col not in self.comments_data.columns:
                print(f"警告: 列 '{col}' 不存在，跳过")
                continue

            has_label = ~self.comments_data[col].isnull()
            
            if has_label.sum() == 0:
                print(f"警告: 列 '{col}' 没有标签数据，跳过")
                continue
                
            print(f"  正在处理 {col}...")

            self.predictors[col] = make_pipeline(
                TfidfVectorizer(tokenizer=jieba.lcut),
                SGDClassifier(random_state=42)
            )

            self.predictors[col].fit(
                self.comments_data[has_label]["comment_text"],
                self.comments_data[has_label][col]
            )

            self.comments_data[col] = self.predictors[col].predict(
                self.comments_data["comment_text"]
            )
            
    def cluster_comments_by_sentiment(self):
        """根据情感分类聚类评论"""
        print("正在分析情感主题...")

        self._cluster_and_extract_themes(
            condition=self.comments_data["sentiment_category"].isin([1, 3]),
            theme_column="positive_cluster_theme",
            description="积极情感"
        )

        self._cluster_and_extract_themes(
            condition=self.comments_data["sentiment_category"].isin([2, 3]),
            theme_column="negative_cluster_theme",
            description="消极情感"
        )
        
    def cluster_comments_by_type(self):
        """根据评论类型聚类"""
        print("正在分析评论类型主题...")

        self._cluster_and_extract_themes(
            condition=self.comments_data["user_scenario"].isin([1]),
            theme_column="scenario_cluster_theme",
            description="使用场景"
        )

        self._cluster_and_extract_themes(
            condition=self.comments_data["user_question"].isin([1]),
            theme_column="question_cluster_theme",
            description="用户问题"
        )

        self._cluster_and_extract_themes(
            condition=self.comments_data["user_suggestion"].isin([1]),
            theme_column="suggestion_cluster_theme",
            description="用户建议"
        )
        
    def _cluster_and_extract_themes(self, condition, theme_column, description):
        """聚类分析并提取主题词"""

        filtered_comments = self.comments_data[condition]["comment_text"]
        
        if len(filtered_comments) == 0:
            print(f"  警告: {description} 没有符合条件的评论")
            return
            
        print(f"  正在分析 {description} ({len(filtered_comments)} 条评论)...")

        kmeans_predictor = make_pipeline(
            TfidfVectorizer(tokenizer=jieba.lcut),
            KMeans(n_clusters=2, random_state=42)
        )
 
        kmeans_predictor.fit(filtered_comments)

        cluster_labels = kmeans_predictor.predict(filtered_comments)
   
        themes = self._extract_cluster_themes(kmeans_predictor)

        self.comments_data.loc[condition, theme_column] = [
            themes[label] for label in cluster_labels
        ]
        
    def _extract_cluster_themes(self, kmeans_predictor):
        """提取聚类主题词"""
        tfidf_vectorizer = kmeans_predictor.named_steps['tfidfvectorizer']
        kmeans_model = kmeans_predictor.named_steps['kmeans']
        
        feature_names = tfidf_vectorizer.get_feature_names_out()
        cluster_centers = kmeans_model.cluster_centers_
        
        themes = []
        for i in range(kmeans_model.n_clusters):
            top_feature_indices = cluster_centers[i].argsort()[::-1]
            top_words = [feature_names[idx] for idx in top_feature_indices[:self.top_n_words]]
            theme = ' '.join(top_words)
            themes.append(theme)
            
        return themes
        
    def run_analysis(self):
        """运行完整分析流程"""
        print("=" * 50)
        print("开始视频评论分析")
        print("=" * 50)

        self.preprocess_video_data()
  
        self.predict_product_names()

        self.predict_comment_categories()

        self.cluster_comments_by_sentiment()
        self.cluster_comments_by_type()
        
        print("=" * 50)
        print("分析完成!")
        print("=" * 50)
        
    def get_analysis_summary(self):
        """获取分析摘要"""
        print("\n分析结果摘要:")
        print("-" * 30)

        if self.video_data is not None:
            print(f"视频总数: {len(self.video_data)}")
            if 'product_name' in self.video_data.columns:
                unique_products = self.video_data['product_name'].nunique()
                print(f"识别出的产品种类: {unique_products}")

        if self.comments_data is not None:
            print(f"评论总数: {len(self.comments_data)}")
  
            if 'sentiment_category' in self.comments_data.columns:
                sentiment_counts = self.comments_data['sentiment_category'].value_counts()
                print(f"情感分布: {dict(sentiment_counts)}")

            theme_columns = ['positive_cluster_theme', 'negative_cluster_theme', 
                           'scenario_cluster_theme', 'question_cluster_theme', 
                           'suggestion_cluster_theme']
            
            for col in theme_columns:
                if col in self.comments_data.columns:
                    theme_count = self.comments_data[col].notna().sum()
                    print(f"{col}: {theme_count} 条评论已分析")

# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = VideoCommentAnalyzer(top_n_words=10)
    
    # 加载数据
    analyzer.load_data()
    
    # 运行分析
    analyzer.run_analysis()
    
    # 显示摘要
    analyzer.get_analysis_summary()
    
    # 可以访问处理后的数据
    print(analyzer.video_data.head())
    print(analyzer.comments_data.head())