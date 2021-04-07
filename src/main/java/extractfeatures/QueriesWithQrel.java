package extractfeatures;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class QueriesWithQrel {

    private static void filter(String original, String relevants, String output) {
        final List<String> qrels = new ArrayList<>();
        try (Scanner sc = new Scanner(new File(relevants))) {
            while (sc.hasNextLine()) {
                final String line = sc.nextLine();
                if (line == null || line.equals(""))
                    continue;
                qrels.add(line.split("\t")[0].trim());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        try (BufferedReader br = new BufferedReader(new FileReader(original))) {
            try (PrintWriter w = new PrintWriter(output)) {
                String line = br.readLine();
                while(line != null) {
                    String queryID = line.split("\t")[0].trim();

                    if (qrels.contains(queryID)) {
                        w.println(line);
                    }

                    line = br.readLine();
                }
            }
            catch(Exception e) {
                e.printStackTrace();
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        if (args.length == 3) {
            filter(args[0], args[1], args[2]);
        } else if (args.length == 2) {
            filter(args[0], args[1], "filtered_features.txt");
        } else {
            System.out.println("Please provide at least 2 arguments: (1) the original query file and (2) the " +
                    "qrel file (and optional: (3) the output file (default 'filtered_queries.tsv'))");
        }
    }
}
