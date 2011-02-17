package edu.berkeley.nlp.assignments.assign2.student;

import java.util.ArrayList;
import java.util.List;
import java.util.Comparator;
import java.util.Collections;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DecoderFactory;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;
import edu.berkeley.nlp.util.CollectionUtils;

public class MonotonicWithLmDecoderFactory implements DecoderFactory
{
    static final Comparator<TranslationWithEndingContext> compareTranslations =
	new Comparator<TranslationWithEndingContext>() {
	public int compare(TranslationWithEndingContext t1, TranslationWithEndingContext t2) {
	    return (t1.lmScore + t1.tmScore) > (t2.lmScore + t2.tmScore) ? -1 : 1;
	}
    };

    class TranslationWithEndingContext {
	int endingContext, currLmContextLength;
	double tmScore, lmScore;
	List<ScoredPhrasePairForSentence> translation;
	int[] lmContextBuf;

	TranslationWithEndingContext(int endingContext, double tmScore, double lmScore,
				     List<ScoredPhrasePairForSentence> translation,
				     int[] lmContextBuf, int currLmContextLength) {
	    this.currLmContextLength = currLmContextLength;
	    this.endingContext = endingContext;
	    this.tmScore = tmScore;
	    this.lmScore = lmScore;
	    this.translation = translation;
	    this.lmContextBuf = lmContextBuf;
	}

	TranslationWithEndingContext addMoreTranslation(int endingContext, 
							ScoredPhrasePairForSentence sppfs,
							PhraseTableForSentence tmState,							
							NgramLanguageModel lm,
							boolean end) {

	    int[] newLmContextBuf = new int[lmContextBuf.length];
	    int newCurrLmContextLength = currLmContextLength;
	    
	    System.arraycopy(lmContextBuf, 0, newLmContextBuf, 0, currLmContextLength);

	    if (newCurrLmContextLength + sppfs.english.indexedEnglish.length >= newLmContextBuf.length)
		newLmContextBuf = CollectionUtils.copyOf(newLmContextBuf, newCurrLmContextLength + sppfs.english.indexedEnglish.length + 1);

	    System.arraycopy(sppfs.english.indexedEnglish, 0, newLmContextBuf, newCurrLmContextLength,
			     sppfs.english.indexedEnglish.length);
	    int currTrgLength = newCurrLmContextLength + sppfs.english.indexedEnglish.length;
	    if (end) {
		newLmContextBuf[currTrgLength] = EnglishWordIndexer.getIndexer().addAndGetIndex(NgramLanguageModel.STOP);
		currTrgLength++;
	    }
	    double lmScore = scoreLm(lm.getOrder(), newCurrLmContextLength, newLmContextBuf, currTrgLength, lm);
	    newCurrLmContextLength += sppfs.english.indexedEnglish.length;

	    List<ScoredPhrasePairForSentence> newList = ((List)((ArrayList)translation).clone());
	    newList.add(sppfs);

	    if (newCurrLmContextLength >= lm.getOrder()) {
		System.arraycopy(newLmContextBuf, newCurrLmContextLength - lm.getOrder() + 1, newLmContextBuf, 0, lm.getOrder() - 1);
		newCurrLmContextLength = lm.getOrder() - 1;
	    }

	    return new TranslationWithEndingContext(endingContext, this.tmScore + sppfs.score, this.lmScore + lmScore, 
						    newList, newLmContextBuf, newCurrLmContextLength);
	}


	private double scoreLm(final int lmOrder, final int prevLmStateLength, 
			       final int[] lmStateBuf, final int totalTrgLength, final NgramLanguageModel lm) {
	    double score = 0.0;
	    
	    if (prevLmStateLength < lmOrder - 1) {
		for (int i = 1; prevLmStateLength + i < lmOrder; ++i) {
		    final double lmProb = lm.getNgramLogProbability(lmStateBuf, 0, prevLmStateLength + i);
		    score += lmProb;
		}
	    }
	    for (int i = 0; i <= totalTrgLength - lmOrder; ++i) {
		final double lmProb = lm.getNgramLogProbability(lmStateBuf, i, i + lmOrder);
		score += lmProb;
	    }
	    return score;
	}

	private double scoreLm(int[] lmStateBuf, int length, NgramLanguageModel lm) {
	    double score = 0.0;
	    for (int i = 1; i < lm.getOrder() + 1 && i < length; i ++) {
		score += lm.getNgramLogProbability(lmStateBuf, 0, i);
	    }

	    for (int i = 1; i < length - lm.getOrder(); i ++) {
		score += lm.getNgramLogProbability(lmStateBuf, i, i + lm.getOrder());
	    }
	    return score;
	}
    }
    
    public class MonotonicWithLmDecoder implements Decoder {
	private PhraseTable tm;
	private NgramLanguageModel lm;
	private DistortionModel dm;
	
	public final int BEAM_SIZE = 2000;
	
	public MonotonicWithLmDecoder(PhraseTable tm, NgramLanguageModel lm, 
				    DistortionModel dm) {
	    super();
	    this.tm = tm;
	    this.lm = lm;
	    this.dm = dm;
	}
	
	private List<TranslationWithEndingContext> getMoreMonotonicTranslations(TranslationWithEndingContext t,
										PhraseTableForSentence tmState, 
										NgramLanguageModel lm, 
										List<String> sentence) {
	    List<TranslationWithEndingContext> beam = new ArrayList<TranslationWithEndingContext>();
	    for (int i = t.endingContext + 1; i < t.endingContext + tmState.getMaxPhraseLength(); i++) {
		List<ScoredPhrasePairForSentence> translations = 
		    tmState.getScoreSortedTranslationsForSpan(t.endingContext, i);
		if (translations == null) {
		    continue;
		}

		for (ScoredPhrasePairForSentence translation : translations) {
		    beam.add(t.addMoreTranslation(i, translation, tmState, lm, i == sentence.size()));
		}
	    }
	    return beam;
	}

	public List<ScoredPhrasePairForSentence> decode(List<String> sentence) {
	    List<TranslationWithEndingContext> beam;
	    PhraseTableForSentence tmState = tm.initialize(sentence);
	    int[] lmContextBuf = new int[sentence.size()];
	    lmContextBuf[0] = EnglishWordIndexer.getIndexer().addAndGetIndex(NgramLanguageModel.START);

	    beam = getMoreMonotonicTranslations(new TranslationWithEndingContext(0, 0.0, 0.0,
										 new ArrayList<ScoredPhrasePairForSentence>(),
										 lmContextBuf, 1),
						tmState, this.lm, sentence);

	    List<TranslationWithEndingContext> doneBeam = new ArrayList<TranslationWithEndingContext>();
	    while (true) {
		List<TranslationWithEndingContext> newBeam = new ArrayList<TranslationWithEndingContext>();

		boolean newTranslationsAdded = false;
		for (TranslationWithEndingContext translation: beam) {
			List<TranslationWithEndingContext> moreTranslations = 
			    getMoreMonotonicTranslations(translation, tmState, this.lm, sentence);

			for (TranslationWithEndingContext newTranslation : moreTranslations) {
			    if (newTranslation.endingContext == sentence.size())
				//doneBeam.add(newTranslation);
				return newTranslation.translation;
			    else {
				newBeam.add(newTranslation);
				newTranslationsAdded = true;
			    }
			}
			if (newBeam.size() > 1 * BEAM_SIZE)
			    break;

		}

		if (!newTranslationsAdded) {
		    Collections.sort(doneBeam, MonotonicWithLmDecoderFactory.compareTranslations);
		    return doneBeam.get(0).translation;
		}

		Collections.sort(newBeam, MonotonicWithLmDecoderFactory.compareTranslations);
		beam = new ArrayList<TranslationWithEndingContext>();
		for (int i = 0; i < BEAM_SIZE && i < newBeam.size(); i++) {
		    TranslationWithEndingContext t = newBeam.get(i);		 
		    /*
		    if (t.endingContext == sentence.size()) {
			return t.translation;
		    }
		    */
		    beam.add(t);
		}
	    }
	}
    }

    public Decoder newDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
	return new MonotonicWithLmDecoder(tm, lm, dm);
    }
}
