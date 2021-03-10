package extractfeatures;

import index.BuildIndex;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class QueryFeatures {
    private static void qf(String queries, String features, String newFile, boolean rmStopwords) throws IOException {
        final BufferedReader brQueries = new BufferedReader(new FileReader(queries));
        List<String> stopWords = null;
        if (rmStopwords) {
            System.out.println("Loading stopwords...");
            stopWords = Arrays.asList(BuildIndex.getStopWords());
        }

        System.out.println("Loading queries...");
        HashMap<String, Integer> queryMap = new HashMap<>();

        String curQuery = brQueries.readLine();
        while (curQuery != null) {
            queryMap.put(curQuery.split(";")[0], getFeatureValue(curQuery.split(";")[1], stopWords));
            curQuery = brQueries.readLine();
        }


        System.out.println("Loading filesize...");
        BufferedReader reader = new BufferedReader(new FileReader(features));
        int lineCount = 0;
        while (reader.readLine() != null) lineCount++;
        reader.close();

        System.out.println("Starting feature generator...");
        int i = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(features))) {
            try (PrintWriter w = new PrintWriter(newFile)) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    String qid = values[1];
                    w.println(line + "," + queryMap.get(qid));
                    if (i % 10000 == 0) {
                        System.out.println(i + "/" + lineCount);
                    }
                    i++;
                }
                w.close();
                System.out.println("Done");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static int getFeatureValue(String query, List<String> removeStopWords) throws FileNotFoundException {
        int wordCount;
        if (removeStopWords != null) {
            List<String> stopWords = Arrays.asList(BuildIndex.getStopWords());
            String[] indivWords = query.split(" ");
            List<String> res = new ArrayList<>();
            for (String indivWord : indivWords) {
                if (!stopWords.contains(indivWord)) {
                    res.add(indivWord);
                }
            }
            wordCount = res.size();
        } else {
            wordCount = query.split(" ").length;
        }
        return wordCount;
    }

    public static void main(String[] args) throws IOException {
        if (args.length == 3) {
            qf(args[1], args[2], "output_plus_qf.csv", args[0].equals("remove"));
        } else if (args.length == 4) {
            qf(args[1], args[2], args[3],args[0].equals("remove"));
        } else {
            System.out.println("Please provide at least 3 arguments: (1) remove (or keep) to (not) remove stopwords " +
                    "in the count,  (2) the (formatted) queries file and (3) the features fxt csv file (Optional: " +
                    "(3) the output file (default 'output_plus_qf.csv'))");
        }
    }
}
