# text-classification

文本分类，使用搜狗新闻语料。

数据源：[搜狗实验室](http://download.labs.sogou.com/resource/ca.php)

使用前，要将数据转码为 utf-8 格式，避免乱码：

```
iconv -f gbk -t utf-8 -c news_tensite_xml.dat > news.dat
```

将全部数据保存到数据库中，方便存取。


类说明：

- MysqlConnector: mysql 数据库操作
- NewsCategorizer: 新闻标注，根据网址
- Segmentation: 新闻分词， 使用 jieba
- SogouDataExtractor: 从数据库中提取固定量的训练、验证与测试数据
- SogouNewsFromFile: 从文件中读取新闻存库
- SogouNewsParser: 逐条解析新闻 xml 文件
- TextPreprocessing: 文本预处理
- XmlParser: 解析整个 xml 文件
