use self::file_db::FileDbWriter;
use ahash::HashMap;
use anyhow::Result;
use file_db::FileDbReader;
use genanki_rs::{Deck, Field, Model, ModelType, Note, Template};
use indexical::{define_index_type, map::DenseRefIndexMap, IndexedDomain};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use jieba_rs::Jieba;
use rand::{seq::SliceRandom, thread_rng};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use std::{
  fs::File,
  io::{BufRead, BufReader},
  ops::Range,
  sync::LazyLock,
};
use unicode_segmentation::UnicodeSegmentation;

mod file_db;

#[derive(PartialEq, Debug, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
struct HskLevel(usize);

fn parse_level<'de, D>(deserializer: D) -> Result<HskLevel, D::Error>
where
  D: Deserializer<'de>,
{
  let buf = String::deserialize(deserializer)?;
  if buf == "7-9" {
    Ok(HskLevel(7))
  } else {
    match buf.parse::<usize>() {
      Ok(n) => Ok(HskLevel(n)),
      Err(e) => Err(serde::de::Error::custom(e)),
    }
  }
}

#[derive(Deserialize, Hash, PartialEq, Eq, Clone)]
struct HskPhrase {
  #[serde(rename = "Simplified")]
  simplified: String,
  #[serde(rename = "Level", deserialize_with = "parse_level")]
  level: HskLevel,
}

fn hsk_levels() -> impl DoubleEndedIterator<Item = HskLevel> {
  (1..=7).map(HskLevel)
}

define_index_type! {
  struct PhraseIdx for HskPhrase = u16;
}

#[derive(Deserialize)]
struct CorpusEntry {
  text: String,
  score: f64,
}

const HSK_PATH: &str = "../hsk30-expanded.csv";

struct Hsk {
  phrases: IndexedDomain<HskPhrase>,
  levels: HashMap<HskLevel, HashMap<String, PhraseIdx>>,
}

fn read_hsk() -> Result<Hsk> {
  let file = BufReader::new(File::open(HSK_PATH)?);
  let mut rdr = csv::Reader::from_reader(file);
  let phrases = rdr
    .deserialize()
    .map(|r| r.map_err(Into::into))
    .collect::<Result<Vec<HskPhrase>>>()?;
  let phrases = IndexedDomain::from_iter(phrases);
  let levels = hsk_levels()
    .map(|level| {
      let level_phrases = phrases
        .iter_enumerated()
        .filter(|(_, phrase)| phrase.level == level)
        .map(|(idx, phrase)| (phrase.simplified.clone(), idx))
        .collect::<HashMap<_, _>>();
      (level, level_phrases)
    })
    .collect::<HashMap<_, _>>();
  Ok(Hsk { phrases, levels })
}

const CORPUS_PATHS: &[&str] = &[
  "../corpus/part-0000.jsonl",
  "../corpus/part-0001.jsonl",
  "../corpus/part-0002.jsonl",
  "../corpus/part-0003.jsonl",
  "../corpus/part-0004.jsonl",
  "../corpus/part-0005.jsonl",
];

const PHRASES_PATH: &str = "../phrases.txt";

fn progress_bar(count: usize) -> ProgressBar {
  ProgressBar::new(count as u64).with_style(
    ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {eta}")
      .unwrap(),
  )
}

fn split_sentences(text: &'_ str) -> Vec<&'_ str> {
  static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[。！？]+").unwrap());
  RE.split(text)
    .filter(|s| !s.is_empty())
    .map(|s| s.trim())
    .collect()
}

const SCORE_THRESHOLD: f64 = 0.8;
const LEN_THRESHOLD: usize = 10;

impl Hsk {
  fn analyze<'a>(&'a self, words: &'a [&str]) -> Option<impl Iterator<Item = PhraseIdx> + 'a> {
    let word_analysis = words
      .iter()
      .map(|word| {
        hsk_levels()
          .rev()
          .find_map(|level| Some((level, *self.levels[&level].get(*word)?)))
      })
      .collect::<Option<Vec<_>>>()?;
    let level = *word_analysis.iter().map(|(level, _)| level).max()?;
    Some(
      word_analysis
        .into_iter()
        .map(|(_, idx)| idx)
        .filter(move |idx| self.phrases.value(*idx).level == level),
    )
  }
}

#[derive(Deserialize, Serialize)]
struct Snippet {
  prefix: Option<String>,
  sentence: String,
  suffix: Option<String>,
}

type PhraseFileIndex<'a> = DenseRefIndexMap<'a, HskPhrase, Vec<Range<u64>>>;

fn build_corpus(hsk: &'_ Hsk) -> Result<PhraseFileIndex<'_>> {
  let mut phrase_map = DenseRefIndexMap::new(&&hsk.phrases, |_| Vec::<Range<u64>>::default());

  let mut db_writer = FileDbWriter::new(PHRASES_PATH)?;

  let segmenter = Jieba::new();
  for path in CORPUS_PATHS
    .iter()
    .progress_with(progress_bar(CORPUS_PATHS.len()))
  {
    let file = BufReader::new(File::open(path)?);

    for line_res in file.lines().take(100000) {
      let line = line_res?;
      let entry: CorpusEntry = serde_json::from_str(&line)?;
      if entry.score < SCORE_THRESHOLD {
        continue;
      }

      let text = html_escape::encode_safe(&entry.text);
      let sentences = split_sentences(text.as_ref());

      let sentence_analysis = sentences
        .into_iter()
        .map(|sentence| {
          let words = segmenter.cut(sentence, false);
          let phrases = hsk.analyze(&words)?.collect::<Vec<_>>();
          Some((sentence, phrases))
        })
        .collect::<Vec<_>>();

      for i in 0..sentence_analysis.len() {
        let Some((sentence, phrases)) = &sentence_analysis[i] else {
          continue;
        };

        if sentence.graphemes(true).count() < LEN_THRESHOLD {
          continue;
        }

        let prefix = if i > 0 {
          sentence_analysis[i - 1]
            .as_ref()
            .map(|(sentence, _)| sentence.to_string())
        } else {
          None
        };

        let suffix = if i < sentence_analysis.len() - 1 {
          sentence_analysis[i + 1]
            .as_ref()
            .map(|(sentence, _)| sentence.to_string())
        } else {
          None
        };

        let snippet = Snippet {
          sentence: sentence.to_string(),
          prefix,
          suffix,
        };

        let range = db_writer.write(&snippet)?;

        for idx in phrases {
          phrase_map[*idx].push(range.clone());
        }
      }
    }
  }

  for idx in hsk.phrases.indices() {
    phrase_map[idx].dedup();
  }

  Ok(phrase_map)
}

const MODEL_ID: i64 = 1122338855;

static MODEL: LazyLock<Model> = LazyLock::new(|| {
  Model::new_with_options(
    MODEL_ID,
    "Cloze (zhlearn)",
    vec![
        Field::new("Sentence"),
        Field::new("Prefix"),
        Field::new("Suffix"),
    ],
    vec![
        Template::new("Cloze")
            .qfmt("<div class=context>{{Prefix}}</div> {{cloze:Sentence}} <div class=context>{{Suffix}}</div>")
            .afmt("<div class=context>{{Prefix}}</div> {{cloze:Sentence}} <div class=context>{{Suffix}}</div>"),
    ],
    Some(
        r#"
.card {
  font-family: arial;
  font-size: 24px;
  text-align: center;
  color: black;
  background-color: white;
}

.context {
  font-size: 80%;
  padding: 0.5rem 0;
}

.cloze {
  font-weight: bold;
  color: blue;
}

.nightMode .cloze {color: lightblue;}"#,
    ),
    Some(ModelType::Cloze),
    None,
    None,
    None,
  )
});

fn make_cloze(sentence: &str, phrase: &str, loc: usize) -> String {
  let mut sentence = sentence.to_string();
  let hole = format!("{{{{c1::{phrase}}}}}");
  sentence.replace_range(loc..loc + phrase.len(), &hole);
  sentence
}

fn build_card(snippet: &Snippet, phrase: &str) -> Note {
  let sentences = split_sentences(&snippet.sentence);
  let (i, loc) = sentences
    .iter()
    .enumerate()
    .find_map(|(i, s)| Some((i, s.find(phrase)?)))
    .unwrap();
  let cloze = make_cloze(sentences[i], phrase, loc);
  Note::new(
    MODEL.clone(),
    vec![
      &cloze,
      snippet.prefix.as_deref().unwrap_or(""),
      snippet.suffix.as_deref().unwrap_or(""),
    ],
  )
  .unwrap()
}

const DECK_ID_BASE: usize = 881199;

fn build_decks(hsk: &Hsk, file_index: &PhraseFileIndex) -> Result<()> {
  let mut reader = FileDbReader::load(PHRASES_PATH)?;

  for level in hsk_levels().progress_with(progress_bar(7)) {
    let phrase_iter = hsk
      .phrases
      .iter_enumerated()
      .filter(|(_, phrase)| phrase.level == level);

    let mut snippets = phrase_iter
      .flat_map(|(phrase_idx, phrase)| {
        file_index[phrase_idx]
          .iter()
          .map(|range| (reader.read::<Snippet>(range.clone()).unwrap(), phrase))
          .collect::<Vec<_>>()
      })
      .collect::<Vec<_>>();

    snippets.shuffle(&mut thread_rng());

    snippets.sort_by_key(|(snippet, _)| {
      let mut score = 0;
      if snippet.prefix.is_some() {
        score += 1;
      }
      if snippet.suffix.is_some() {
        score += 1;
      }
      -score
    });

    let mut deck = Deck::new(
      (DECK_ID_BASE + level.0) as i64,
      &format!("HSK Level {}", level.0),
      "Corpus-generated Chinese Cloze cards",
    );
    for (snippet, phrase) in snippets.into_iter().take(50) {
      let note = build_card(&snippet, &phrase.simplified);
      deck.add_note(note);
    }

    deck.write_to_file(&format!("../decks/hsk-{}.apkg", level.0))?;
  }

  Ok(())
}

fn main() -> Result<()> {
  let hsk = &read_hsk()?;
  let file_index = &build_corpus(hsk)?;

  for level in hsk_levels() {
    let phrase_iter = hsk
      .phrases
      .iter_enumerated()
      .filter(|(_, phrase)| phrase.level == level);
    println!(
      "{level:?}: {}",
      phrase_iter
        .map(|(idx, _)| file_index[idx].len())
        .sum::<usize>()
    );
  }

  build_decks(hsk, file_index)?;
  Ok(())
}
