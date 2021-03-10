package evaluate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Scanner;

public class CombineQueryText {

    public static void findQuery(String scores, String queries, String output) {
        final HashMap<String, String> qs = new HashMap<>();
        try (Scanner sc = new Scanner(new File(queries))) {
            while (sc.hasNextLine()) {
                final String[] line = sc.nextLine().split("\t");
                if (line == null || line.equals(""))
                    continue;
                qs.put(line[0], line[1]);
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        try (BufferedReader br = new BufferedReader(new FileReader(scores))) {
            try (PrintWriter w = new PrintWriter(output)) {
                String line = br.readLine();
                while(line != null) {
                    String queryID = line.split(";")[1].trim();

                    w.println(line + ";" + qs.get(queryID));

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
            findQuery(args[0], args[1], args[2]);
        } else if (args.length == 2) {
            findQuery(args[0], args[1], "queries_results.csv");
        } else {
            System.out.println("Please provide at least 2 arguments: (1) the scores csv file and (2) the " +
                    "tsv file with queries (and optional: " +
                    "(3) the output file (default 'queries_results.csv'))");
        }
    }
}
