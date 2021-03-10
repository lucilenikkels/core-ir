package evaluate;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ScoreToTrec {
    public static void main(String[] args) {

    }

    private void toTrec(String scoreFile, String docFile, String newFile) {
        HashMap<Float, String> docs = new HashMap<>();
        String prev = "";

        try {
            final BufferedReader brScore = new BufferedReader(new FileReader(scoreFile));
            final BufferedReader brDoc = new BufferedReader(new FileReader(docFile));
            final PrintWriter w = new PrintWriter(newFile);

            String scLine = brScore.readLine();
            String dcLine = brDoc.readLine();

            while(scLine != null && dcLine != null) {
                String[] score = scLine.split(" ");
                String[] doc = dcLine.split(" ");

                if (!score[0].equals(prev)) {

                }

                scLine = brScore.readLine();
                dcLine = brDoc.readLine();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
