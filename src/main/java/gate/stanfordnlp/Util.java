package gate.stanfordnlp;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Util {

	public static String[] stringToArgs(String string) {
		List<String> args = new ArrayList<>();

		String regex = "\"([^\"]*)\"|(\\S+)";
		Matcher m = Pattern.compile(regex).matcher(string);
		while (m.find()) {
			if (m.group(1) != null) {
				args.add(m.group(1));
			} else {
				args.add(m.group(2));
			}
		}

		return args.toArray(new String[] {});
	}

}
