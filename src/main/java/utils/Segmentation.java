package utils;

import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.SegToken;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by gaussic on 2017/3/20.
 * 分词工具，暂用 jieba
 */
public class Segmentation {

    private JiebaSegmenter segmenter = new JiebaSegmenter();

    public List<String> getSegmented(String sentence) {
        List<SegToken> segmented = segmenter.process(sentence, JiebaSegmenter.SegMode.SEARCH);
        List<String> tokens = new ArrayList<>();
        for (SegToken s : segmented) tokens.add(s.word);
        return tokens;
    }

    public static void main(String[] args) {

        Segmentation segmenter = new Segmentation();
        String[] sentences = new String[]{"这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。",
                "我不喜欢日本和服。", "雷猴回归人间。", "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作",
                "结果婚的和尚未结过婚的"};

        for (String sentence : sentences) {
            System.out.println(segmenter.getSegmented(sentence));
        }
    }
}
