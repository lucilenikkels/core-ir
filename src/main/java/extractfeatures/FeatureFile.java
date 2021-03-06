package extractfeatures;

import java.io.*;
import java.util.*;

public class FeatureFile {

    /**
     * output.csv:
     * 0 or 1, relevance label?
     * qid
     * docid
     * 1 f_bm25_atire = 1
     * 2 f_bm25_trec3 = 1
     * 3 f_bm25_trec3_kmax = 1
     * 4 f_lm_dir_2500 = 1
     * 5 f_lm_dir_1500 = 1
     * 6 f_lm_dir_1000 = 1
     * 7 f_tfidf = 1
     * 8 f_prob = 1
     * 9 f_be = 1
     * 10 f_dph = 1
     * 11 f_dfr = 1
     * 12 f_stream_len = 1
     * 13 f_sum_stream_len = 1
     * 14 f_min_stream_len = 1
     * 15 f_max_stream_len = 1
     * 16 f_mean_stream_len = 1
     * 17 f_variance_stream_len = 1
     */
    private static final int NUM_FEATURES = 17;

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length == 0) {
            System.out.println("Please provide 2 file locations: (1) the existing features/output file, " +
                    "(2) the output location/file");
        } else {
            System.out.println("Loading features...");
            try (BufferedReader br = new BufferedReader(new FileReader(args[0]))) {
                try (PrintWriter w = new PrintWriter(args[1])) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] values = line.split(",");
                        w.print(values[0] + " qid:" + values[1]);
                        // Skip value at i=2 because we don't need the doc id
                        for (int i = 3; i < NUM_FEATURES+3; i++) {
                            w.print(" " + (i-2) + ":" +  values[i]);
                        }
                        w.print('\n');
                    }
                    System.out.println("Done");
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
