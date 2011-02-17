package edu.berkeley.nlp.assignments.assign2.student;

import java.util.ArrayList;
import java.util.List;
import java.util.Comparator;
import java.util.Collections;

import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.mt.decoder.Decoder;
import edu.berkeley.nlp.mt.decoder.DecoderFactory;
import edu.berkeley.nlp.mt.decoder.DistortionModel;
import edu.berkeley.nlp.mt.phrasetable.PhraseTable;
import edu.berkeley.nlp.mt.phrasetable.PhraseTableForSentence;
import edu.berkeley.nlp.mt.phrasetable.ScoredPhrasePairForSentence;

public class MonotonicNoLmDecoderFactory implements DecoderFactory
{
    static final Comparator<TranslationWithEndingContext> compareTranslations =
	new Comparator<TranslationWithEndingContext>() {
	public int compare(TranslationWithEndingContext t1, TranslationWithEndingContext t2) {
	    return t1.score > t2.score ? -1 : 1;
	}
    };

    class TranslationWithEndingContext {
	int endingContext;
	double score;
	List<ScoredPhrasePairForSentence> translation;

	TranslationWithEndingContext(int endingContext, double score,
				     List<ScoredPhrasePairForSentence> translation) {
	    this.endingContext = endingContext;
	    this.score = score;
	    this.translation = translation;
	}

	TranslationWithEndingContext addMoreTranslation(int endingContext, double score, ScoredPhrasePairForSentence sppfs) {
	    List<ScoredPhrasePairForSentence> newList = ((List)((ArrayList)translation).clone());
	    newList.add(sppfs);
	    return new TranslationWithEndingContext(endingContext, this.score + score, newList);
	}
    }
    
    public class MonotonicNoLmDecoder implements Decoder {
	private PhraseTable tm;
	private DistortionModel dm;
	
	public final int BEAM_SIZE = 2000;
	
	public MonotonicNoLmDecoder(PhraseTable tm, NgramLanguageModel lm, 
				    DistortionModel dm) {
	    super();
	    this.tm = tm;
	    this.dm = dm;
	}
	
	private List<TranslationWithEndingContext> getMoreMonotonicTranslations(TranslationWithEndingContext t,
										PhraseTableForSentence tmState) {
	    List<TranslationWithEndingContext> beam = new ArrayList<TranslationWithEndingContext>();
	    for (int i = t.endingContext + 1; i < t.endingContext + tmState.getMaxPhraseLength(); i++) {
		List<ScoredPhrasePairForSentence> translations = 
		    tmState.getScoreSortedTranslationsForSpan(t.endingContext, i);
		if (translations == null) {
		    continue;
		}

		for (ScoredPhrasePairForSentence translation : translations) {
		    beam.add(t.addMoreTranslation(i, translation.score, translation));
		}
	    }
	    return beam;
	}

	public List<ScoredPhrasePairForSentence> decode(List<String> sentence) {
	    List<TranslationWithEndingContext> beam;
	    PhraseTableForSentence tmState = tm.initialize(sentence);

	    beam = getMoreMonotonicTranslations(new TranslationWithEndingContext(0, 0.0, new ArrayList<ScoredPhrasePairForSentence>()),
					       tmState);

	    List<TranslationWithEndingContext> doneBeam = new ArrayList<TranslationWithEndingContext>();
	    while (true) {
		List<TranslationWithEndingContext> newBeam = new ArrayList<TranslationWithEndingContext>();	       
		boolean newTranslationsAdded = false;

		for (TranslationWithEndingContext translation : beam) {
		    if (translation.endingContext == sentence.size())
			doneBeam.add(translation);
		    else {
			List<TranslationWithEndingContext> moreTranslations = 
			    getMoreMonotonicTranslations(translation, tmState);
			if (moreTranslations.size() > 0)
			    newTranslationsAdded = true;
			
			for (TranslationWithEndingContext newTranslation : moreTranslations) {
			    newBeam.add(newTranslation);
			}
			if (newBeam.size() > BEAM_SIZE)
			    break;
		    }
		}

		if (!newTranslationsAdded) {
		    Collections.sort(doneBeam, MonotonicNoLmDecoderFactory.compareTranslations);		    
		    return doneBeam.get(0).translation;
		}
		
		Collections.sort(newBeam, MonotonicNoLmDecoderFactory.compareTranslations);

		beam = new ArrayList<TranslationWithEndingContext>();
		for (int i = 0; i < BEAM_SIZE && i < newBeam.size(); i++) {
		    beam.add(newBeam.get(i));
		}
	    }
	}

    }

    public Decoder newDecoder(PhraseTable tm, NgramLanguageModel lm, DistortionModel dm) {
	return new MonotonicNoLmDecoder(tm, lm, dm);
    }
}
