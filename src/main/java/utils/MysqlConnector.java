package utils;

import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by gaussic on 2017/3/20.
 * mysql connector to sogou database
 */
public class MysqlConnector {
    private Connection conn = null;
    private Statement stmt = null;
    private PreparedStatement pstmt = null;
    private ResultSet rSet = null;

    // 构造函数创建连接
    public MysqlConnector() {
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
            conn = DriverManager.getConnection(
                    "jdbc:mysql://localhost:3306/sogou_news?useSSL=false&useUnicode=true&characterEncoding=UTF-8 ",
                    "root",
                    "dzkang");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 查看数据库中有多少数据
    public void readNewsCount() {
        try {
            stmt = conn.createStatement();
            rSet = stmt.executeQuery("select count(*) from sogou_news.news");
            while (rSet.next()) {
                System.out.println(rSet.getInt(1));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 插入一条数据
    public void insertNews(Map<String, String> news) {
        try {
            pstmt = conn.prepareStatement("insert into sogou_news.news values (DEFAULT , ?, ?, ?, ?, DEFAULT )");
            pstmt.setString(1, news.get("url"));
            pstmt.setString(2, news.get("docno"));
            pstmt.setString(3, news.get("title"));
            pstmt.setString(4, news.get("content"));
            pstmt.executeUpdate();
        } catch (Exception e) {
            e.printStackTrace();
            //System.out.println("重复");
        }
    }

    // 查询分类标记为其他的数据
    public List<Map<String, String>> findUncategorized(int startIndex, int limit) {
        try {
            pstmt = conn.prepareStatement("select * from sogou_news.news where category = '其他' limit ?, ?");
            pstmt.setInt(1, startIndex);
            pstmt.setInt(1, limit);
            rSet = pstmt.executeQuery();

            List<Map<String, String>> newsList = new ArrayList<>();
            while (rSet.next()) {
                Map<String, String> news = new HashMap<>();
                news.put("id", rSet.getString("id"));
                news.put("url", rSet.getString("url"));
                newsList.add(news);
            }
            return newsList;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // 更新分类
    public void updateCategory(String category, String url, int id) {
        try {
            pstmt = conn.prepareStatement("update sogou_news.news set category = ? where id = ?");
            pstmt.setString(1, category);
            pstmt.setInt(2, id);
            pstmt.executeUpdate();
        } catch (Exception e) {
            System.out.println(url + "   " + category);
        }

    }

    // 文档是否已存在
    public boolean hasDocno(String docno) {
        try {
            pstmt = conn.prepareStatement("select * from sogou_news.news where docno = (?)");
            pstmt.setString(1, docno);
            rSet = pstmt.executeQuery();
            if (rSet.next()) {
                return true;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    // 关闭数据库
    public void closeDatabase() {
        try {
            if (rSet != null) {
                rSet.close();
            }
            if (stmt != null) {
                stmt.close();
            }
            if (pstmt != null) {
                pstmt.close();
            }
            if (conn != null) {
                conn.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        MysqlConnector conn = new MysqlConnector();
        conn.readNewsCount();
        List<Map<String, String>> newsList = conn.findUncategorized(1000, 1000);
        System.out.println(newsList.size());
        for (Map<String, String> news : newsList) {
            System.out.println(news.get("id") + "   " + news.get("url"));
        }
        conn.closeDatabase();
    }


}
