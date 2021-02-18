import lemurproject.indri.*;

public class Main {

    public static void main(String[] args) {
        try {
            String [] stopwords = {"a", "an", "the", "of"};
            IndexEnvironment env = new IndexEnvironment();
            env.setStoreDocs(true);
            env.setStopwords(stopwords);
            env.create("test.idx");
            env.addFile("testText.in", "trectext");
            env.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
