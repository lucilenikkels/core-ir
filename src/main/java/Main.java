import evaluate.CombineQueryText;
import evaluate.QueriesWithQrel;
import extractfeatures.*;
import index.BuildDocumentIdMap;
import index.BuildIndex;
import languagemodel.BuildPrior;
import languagemodel.RunQuery;

import java.util.Arrays;

public class Main {

	public static void main(String[] args) throws Exception {
		if (args.length < 1) {
			System.out.println("Use either rebuild, buildprior or query as program argument,"//
					+ " for example [java -jar tool.jar rebuild]");
			return;
		}
		final String action = args[0];
		if (action.equalsIgnoreCase("rebuild")) {
			BuildIndex.main(new String[0]);
		} else if (action.equalsIgnoreCase("buildprior")) {
			BuildPrior.main(new String[0]);
		} else if (action.equalsIgnoreCase("buildids")) {
			BuildDocumentIdMap.main(new String[0]);
		} else if (action.equalsIgnoreCase("query")) {
			RunQuery.main(new String[0]);
		} else if (action.equalsIgnoreCase("preprocessqueries")) {
			PreprocessQuery.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("csvtofeatures")) {
			FeatureFile.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("balancefeatures")) {
			BalanceDataset.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("combinelabels")) {
			LabelCombine.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("queryfeature")) {
			QueryFeatures.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("positivequeries")) {
			QueriesWithQrel.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("addtext")) {
			CombineQueryText.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("filterqueries")) {
			extractfeatures.QueriesWithQrel.main(Arrays.copyOfRange(args, 1, args.length));
		} else if (action.equalsIgnoreCase("embeddingsfeatures")) {
			extractfeatures.EmbeddingsFeatures.main(Arrays.copyOfRange(args, 1, args.length));
		} else
			System.out.println("Unknown action " + action);
	}
}
