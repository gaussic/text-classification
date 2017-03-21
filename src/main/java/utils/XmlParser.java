package utils;

import entity.News;
import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Created by gaussic on 2017/3/20.
 * 数据量太大，不实用
 * 搜狗的语料xml文件没有根结点，要自己加上
 * 然后 out of memory 了
 */
public class XmlParser {

    // 解析整个文件
    public List<News> parseXml(InputStream cntr) {
        List<News> newsList = new ArrayList<>();
        try {
            SAXReader saxReader = new SAXReader();
            Document document = saxReader.read(cntr);
            Element root = document.getRootElement();
            System.out.println("Root: " + root.getName());

            List<Element> childList = root.elements();

            System.out.println("Total child count: " + childList.size());
            for (Element child : childList) {
                try {
                    String url = child.elementText("url");
                    String docno = child.elementText("docno");
                    String title = child.elementText("contenttitle");
                    String content = child.elementText("content");
                    newsList.add(new News(url, docno, title, content));
                } catch (Exception e) {
                    System.out.println("解析失败。");
                }
            }
        } catch (DocumentException e) {
            e.printStackTrace();
        }

        return newsList;
    }

    public static void main(String[] args) throws Exception {
        // 读取和预处理
        File file = new File("news.dat");
        FileInputStream fis = new FileInputStream(file);
        System.out.println(file.length());
        byte[] data = new byte[(int) file.length()];
        int i = fis.read(data);
        String str = new String(data, "UTF-8").replaceAll("&", "&amp;");

        // 添加根节点，重新转为 InputStream
        List<InputStream> streams = Arrays.asList(new ByteArrayInputStream("<root>".getBytes()),
                new ByteArrayInputStream(str.getBytes()),
                new ByteArrayInputStream("</root>".getBytes()));
        InputStream cntr = new SequenceInputStream(Collections.enumeration(streams));

        XmlParser xmlParser = new XmlParser();
        List<News> newsList = xmlParser.parseXml(cntr);
        System.out.println(newsList.size());
    }
}
