package gate.stanfordnlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Util {

	public static int[] toIntArray(List<Integer> list) {
		int[] ret = new int[list.size()];
		int i = 0;
		for (Integer e : list) {
			ret[i++] = e.intValue();
		}
		return ret;
	}

	public static boolean isAssignableFromAny(Class<?> clazz, Class<?>... others) {
		for (Class<?> class1 : others) {
			if (clazz.isAssignableFrom(class1))
				return true;
		}
		return false;
	}

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

	public static <K, V> void addToListMap(Map<K, List<V>> map, K key, V value) {
		List<V> list = map.get(key);
		if (list != null) {
			list.add(value);
		} else {
			list = new ArrayList<>();
			list.add(value);
			map.put(key, list);
		}
	}
}
