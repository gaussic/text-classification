package utils;

import org.dom4j.Document;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by gaussic on 2017/3/20.
 * 解析原始的xml数据集，总共七百多万行
 * 每六行为一篇文章，组成一个xml结构
 * 因此按6行进行解析
 */
public class SogouNewsParser {

    // 解析一篇新闻
    public Map<String, String> parseNews(String news) {
        try {
            Map<String, String> newsMap = new HashMap<>();
            InputStream stream = new ByteArrayInputStream(news.getBytes());
            SAXReader saxReader = new SAXReader();
            Document document = saxReader.read(stream);
            Element doc = document.getRootElement();

            String url = doc.elementText("url");
            String docno = doc.elementText("docno");
            String title = doc.elementText("contenttitle");
            String content = doc.elementText("content");

            if(url.equals("") || docno.equals("") || title.equals("") || content.equals(""))
                return null;

            newsMap.put("url", url);
            newsMap.put("docno", docno);
            newsMap.put("title", title);
            newsMap.put("content", content);

            return newsMap;
        } catch (Exception e) {
            System.out.println("解析失败。");
            return null;
        }
    }

    public static void main(String[] args) throws Exception {
        MysqlConnector conn = new MysqlConnector();
        SogouNewsParser parser = new SogouNewsParser();

        // 可以先使用小数据集 news_smarty.xml 测试
        FileInputStream fis = new FileInputStream("news.dat");
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String strLine;
        String newsXml = "";
        int cnt = 0;
        int totalNum = 0;
        while ((strLine = br.readLine()) != null) {
            newsXml += strLine + "\n";
            if (cnt == 5) {
                cnt = 0;
                try {
                    Map<String, String> news = parser.parseNews(newsXml.replaceAll("&", "&amp;"));
                    if(news != null)
                        conn.insertNews(news);
                } catch (Exception e) {
                    System.out.println("插入失败。");
                }
                newsXml = "";
                continue;
            }
            cnt++;
            totalNum ++;
            if(totalNum % 5000 == 0) {
                System.out.println(totalNum);
            }
        }
        conn.readNewsCount();
        conn.closeDatabase();
    }
}
