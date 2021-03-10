package evaluate;

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
                qrels.add(line.trim());
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        try (BufferedReader br = new BufferedReader(new FileReader(original))) {
            try (PrintWriter w = new PrintWriter(output)) {
                String line = br.readLine();
                while(line != null) {
                    String queryID = line.split(" ")[1].substring(4);

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
            System.out.println("Please provide at least 2 arguments: (1) the original feature file and (2) the " +
                    "file with query ids that have at least 1 positive sample (use command: \n " +
                    "grep '1 qid' test_features.txt | grep -o -P '(?<=qid:).*(?= 1:)' | sort -u > qrel_test_queries.txt" +
                    " \n) (and optional: " +
                    "(3) the output file (default 'filtered_features.txt'))");
        }
    }
}
