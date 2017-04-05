package utils;

import java.util.List;
import java.util.Map;

/**
 * Created by gaussic on 2017/3/20.
 * 新闻分类，按url
 */
public class NewsCategorizer {

    String[] urls = new String[]{
            "http://auto", "汽车",
            "http://1688.autos", "汽车",
            "http://business", "财经",
            "http://finance", "财经",
            "http://money", "财经",
            "http://biz", "财经",
            "http://1688.tech", "科技",
            "http://taobao.finance", "财经",
            "http://alibaba.finance", "财经",
            "http://it", "科技",
            "http://tech", "科技",
            "http://taobao.diji", "科技",
            "http://health", "健康",
            "http://sina.kangq.com/", "健康",
            "http://fashion.ifeng.com/health/", "健康",
            "http://fashion.ifeng.com/travel/", "旅游",
            "http://tour", "旅游",
            "http://travel", "旅游",
            "http://sports", "体育",
            "http://yundong", "体育",
            "http://taobao.sports", "体育",
            "http://news.ifeng.com/sports/", "体育",
            "http://learning", "教育",
            "http://edu", "教育",
            "http://career", "招聘",
            "http://cul", "文化",
            "http://art", "文化",
            "http://mil", "军事",
            "http://war", "军事",
            "http://news.ifeng.com/mil/", "军事",
            "http://society", "社会",
            "http://news.sohu.com/", "社会",
            "http://news.sina.com.cn/", "社会",
            "http://news.163.com/", "社会",
            "http://news.qq.com/", "社会",
            "http://house", "房产",
            "http://yule", "娱乐",
            "http://ent", "娱乐",
            "http://taobao.ent", "娱乐",
            "http://media", "传媒",
            "http://news.sina.com.cn/media/", "传媒",
            "http://gongyi", "公益",
            "http://women", "时尚",
            "http://eladies", "时尚",
            "http://lady", "时尚",
            "http://luxury", "时尚",
            "http://fashion", "时尚",
            "http://taobao.lady", "时尚"
    };

    public String getCategory(String url) {
        for (int i = 0; i < urls.length; i += 2) {
            if (url.startsWith(urls[i])) {
                return urls[i + 1];
            }
        }
        return "其他";
    }

    public static void main(String[] args) throws Exception {
        NewsCategorizer categorizer = new NewsCategorizer();
        MysqlConnector conn = new MysqlConnector();

        List<Map<String, String>> newsList = conn.findUncategorized(300000, 300);
        for (Map<String, String> news : newsList) {
            String url = news.get("url");
            int id = Integer.parseInt(news.get("id"));
            String cat = categorizer.getCategory(url);
            conn.updateCategory(cat, url, id);
        }
        conn.closeDatabase();

    }


}
