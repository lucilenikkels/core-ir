package extractfeatures;

import index.BuildIndex;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class EmbeddingsFeatures {
    private static void qf(String dists, String features, String newFile) throws IOException {
        final BufferedReader brDists = new BufferedReader(new FileReader(dists));

        System.out.println("Loading queries...");
        HashMap<String, String> featureMap = new HashMap<>();

        String lineFt = brDists.readLine();
        while (lineFt != null) {
            String[] vals = lineFt.split("\t");
            featureMap.put(vals[0]+"_"+vals[1], vals[2]);
            lineFt = brDists.readLine();
        }

        System.out.println("Loading filesize...");
        BufferedReader reader = new BufferedReader(new FileReader(features));
        int lineCount = 0;
        while (reader.readLine() != null) lineCount = lineCount+1;
        reader.close();

        System.out.println("Starting feature generator...");
        int i = 0;
        try (BufferedReader br = new BufferedReader(new FileReader(features))) {
            try (PrintWriter w = new PrintWriter(newFile)) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] values = line.split(",");
                    String key = values[1] + "_" + values[2];
                    String distance = featureMap.get(key);
                    if (distance != null) {
                        w.println(line + "," + distance);
                    } else {
                        System.out.println("Failed for combination "+key);
                    }
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

    public static void main(String[] args) throws IOException {
        if (args.length == 2) {
            qf(args[0], args[1], "output_plus_distance.csv");
        } else if (args.length == 3) {
            qf(args[0], args[1], args[2]);
        } else {
            System.out.println("Please provide at least 2 arguments: (1) the distances file and (2) the " +
                    "features fxt csv file (Optional: (3) the output file (default 'output_plus_distance.csv'))");
        }
    }
}
