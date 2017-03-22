package utils;

import java.io.FileWriter;
import java.util.List;
import java.util.Map;

/**
 * Created by dzkan on 2017/3/22.
 * 随机从数据库中获取数据，存到文件中
 */
public class SogouDataExtractor {
    String[] categories = new String[]{"汽车", "财经", "科技", "健康", "体育", "教育", "文化", "军事", "娱乐", "时尚"};

    public String toString(Map<String, String> news) {
        return news.get("category") + "\t" + news.get("title") + "\t" + news.get("url") + "\t" + news.get("content");
    }

    public void writeDataFile(int trainNum, int valNum, int testNum) {
        try {
            FileWriter fTrain = new FileWriter("train.txt");
            FileWriter fVal = new FileWriter("val.txt");
            FileWriter fTest = new FileWriter("test.txt");
            String newLine = System.getProperty("line.separator");

            int totalNum = trainNum + valNum + testNum;

            MysqlConnector conn = new MysqlConnector();
            for (String category : categories) {
                List<Map<String, String>> newsList = conn.selectRandomNews(category, totalNum);
                for (int j = 0; j < totalNum; j++) {
                    if (j < trainNum) {
                        fTrain.write(toString(newsList.get(j)) + newLine);
                    } else if (j < trainNum + valNum) {
                        fVal.write(toString(newsList.get(j)) + newLine);
                    } else {
                        fTest.write(toString(newsList.get(j)) + newLine);
                    }
                }
            }
            fTrain.close();
            fVal.close();
            fTest.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        SogouDataExtractor extractor = new SogouDataExtractor();
        extractor.writeDataFile(3, 2, 1);
    }
}
