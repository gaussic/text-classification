# text-classification

文本分类，使用搜狗新闻语料。

数据源：[搜狗实验室](http://download.labs.sogou.com/resource/ca.php)

使用前，要将数据转码为 utf-8 格式，避免乱码：

```
iconv -f gbk -t utf-8 -c news_tensite_xml.dat > news.dat
```

将全部数据保存到数据库中，方便存取。