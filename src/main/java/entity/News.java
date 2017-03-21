package entity;

/**
 * Created by gaussic on 2017/3/20.
 * 新闻 POJO
 */
public class News {
    private String url;      // 网址
    private String docno;    // 文档编号
    private String title;    // 标题
    private String content;  // 内容

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getDocno() {
        return docno;
    }

    public void setDocno(String docno) {
        this.docno = docno;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public News(String url, String docno, String title, String content) {
        this.url = url;
        this.docno = docno;
        this.title = title;
        this.content = content;
    }

    public News() {
    }
}
