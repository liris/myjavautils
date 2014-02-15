/*
 Copyright (C) 2010 Hiroki Ohtani(liris)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

package org.liris.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class NaiveBayes {
	Collection<String> vocabularies = new HashSet<>();
	Map<String, Map<String, Integer>> wordCount = new HashMap<>();
	Map<String, Integer> categoryCount = new HashMap<>();
	Map<String, Integer> denominator = new HashMap<>();
	
	public static class Result {
		String category;		
		double score;
		
		private Result(String category, double score) {
			this.category = category;
			this.score = score;
		}
		
		public String getCategory() {
			return category;
		}

		public double getScore() {
			return score;
		}
		
		public String toString() {
			return "category=" + category + " score=" + score;
		}
	}
	
	class ResultCompartor implements Comparator<Result>{
		public int compare(Result r1, Result r2) {
			double d = r2.getScore() - r1.getScore();
			if (d > 0) {
				return 1;
			}
			if (d < 0) {
				return -1;
			}
			return 0;
		}
	}
	
	/**
	 * train to naive bayes.
	 * @param category
	 * @param words
	 */
	public void train(String category, Collection<String> words) {
		increment(categoryCount, category);
		
		Map<String, Integer> counter = wordCount.get(category);
		if (counter == null) {
			counter = new HashMap<String, Integer>();
			wordCount.put(category, counter);
		}
		for (String word: words) {
			vocabularies.add(word);
			increment(counter, word);
		}
		denominator.put(category, vocabularies.size() + sum(counter.values()));
	}
	
	public void train(String category, String[] words) {
		List<String> l = Arrays.asList(words);
		train(category, l);
	}
	
	private int sum(Collection<Integer> data) {
		int total = 0;
		for (int i: data) {
			total += i;
		}
		return total;
	}
	
	private void increment(Map<String, Integer> data, String key) {
		Integer i = data.get(key);
		if (i == null) {
			i = 0;
		}
		data.put(key, i + 1);
	}
		
	/**
	 * clear naive bayes.
	 */
	public void clear() {
		vocabularies.clear();
		wordCount.clear();
		categoryCount.clear();
		denominator.clear();
	}
	
	/**
	 * classify word list and find out the match result.
	 * @param words
	 * @return
	 */
	public List<Result> classify(Collection<String> words) {
		List<Result> result = new ArrayList<>();
		for (String category: categoryCount.keySet()) {
			result.add(new Result(category, score(category, words)));
		}
		Collections.sort(result, new ResultCompartor());
		return result;
	}
	
	private double getWordProbability(String category, String word) {
		Map<String, Integer> counter = wordCount.get(category);
		Integer v = 0;
		if (counter != null) {
			if (counter.containsKey(word)) {
				v = counter.get(word);
			}
		}
		
		return 1.0*(v+1)/denominator.get(category);
	}
	
	private double score(String category, Collection<String> words) {
		int total = sum(categoryCount.values());
		double score = Math.log(1.0*categoryCount.get(category)/total);
		
		for (String word: words) {
			score += Math.log(getWordProbability(category, word));
		}
		
		return score;
	}
	
	public static void main(String[] args) {
		// All test data came from http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf
		NaiveBayes nb = new NaiveBayes();
		nb.train("yes", new String[]{"Chinese", "Beijing", "Chinese"});
		nb.train("yes", new String[]{"Chinese", "Chinese", "Shanghai"});
		nb.train("yes", new String[]{"Chinese", "Monaco"});
		nb.train("no", new String[]{"Tokyo", "Japan", "Chinese"});
		
		System.out.println(nb.getWordProbability("yes", "Chinese"));
		System.out.println(nb.getWordProbability("yes", "Tokyo"));
		System.out.println(nb.getWordProbability("yes", "Japan"));
		System.out.println(nb.getWordProbability("no", "Chinese"));
		System.out.println(nb.getWordProbability("no", "Tokyo"));
		System.out.println(nb.getWordProbability("no", "Japan"));
		
		List<String> testData = Arrays.asList(new String[]{"Chinese", "Chinese", "Chinese", "Tokyo", "Japan"});
		System.out.println(nb.score("yes", testData));
		System.out.println(nb.score("no", testData));
		System.out.println(nb.classify(testData));
	}
}
