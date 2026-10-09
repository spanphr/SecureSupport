import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config import Settings
from app.retrieval.embeddings import OpenAIEmbedder
from app.retrieval.retriever import KnowledgeRetriever
from app.schemas.domain import KnowledgeArticle
from app.schemas.retrieval import RetrievalHit, RetrievalResult
from evaluation.retrieval import RetrievalCase, evaluate, load_cases
from scripts.validate_data import validate_dataset


class FixedEmbedder:
    """Test-only vectors; not a substitute for evaluating semantic embeddings."""
    identity = 'test:fixed:v1'

    def embed(self, texts):
        return [[1.0, 0.0] if 'refund' in text.lower() else [0.0, 1.0] for text in texts]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.articles = [
            KnowledgeArticle(id='kb_refund', title='Refund', content='Request a refund.', tags=['payments']),
            KnowledgeArticle(id='kb_mfa', title='MFA recovery', content='Use verified recovery.', tags=['security']),
        ]
        self.retriever = KnowledgeRetriever(self.articles, FixedEmbedder(), Path(self.folder.name))

    def test_missing_index_and_validation(self):
        with self.assertRaisesRegex(ValueError, 'index is missing'):
            self.retriever.search('refund')
        for query, k in [('', 3), ('x' * 4001, 3), ('refund', 0), ('refund', True)]:
            with self.assertRaises(ValueError):
                self.retriever.search(query, k)

    def test_real_chroma_ranking_persistence_and_idempotence(self):
        self.assertEqual(self.retriever.index(), 2)
        self.assertEqual(self.retriever.search('refund', 10).hits[0].article_id, 'kb_refund')
        reopened = KnowledgeRetriever(self.articles, FixedEmbedder(), Path(self.folder.name))
        with patch.object(reopened.embedder, 'embed', side_effect=AssertionError('Re-embedding')):
            self.assertEqual(reopened.index(), 2)
        self.assertEqual(reopened.search('account recovery').hits[0].article_id, 'kb_mfa')
        changed = self.articles[0].model_copy(update={'content': 'Changed refund policy'})
        newer = KnowledgeRetriever([changed, self.articles[1]], FixedEmbedder(), Path(self.folder.name))
        self.assertFalse(newer.ready)
        self.assertNotEqual(newer.collection.name, reopened.collection.name)

    def test_model_change_isolates_collection(self):
        other = FixedEmbedder()
        other.identity = 'test:other:v1'
        newer = KnowledgeRetriever(self.articles, other, Path(self.folder.name))
        self.assertNotEqual(newer.collection.name, self.retriever.collection.name)

    def test_metric_math_includes_misses(self):
        def result(ids):
            return RetrievalResult(query='q', hits=[RetrievalHit(article_id=i, title=i, content=i, distance=0) for i in ids])
        fake = SimpleNamespace(search=lambda query, top_k: result({'first':['a'], 'third':['x','y','a'], 'miss':['x']}[query]))
        cases = [RetrievalCase(id=q, query=q, relevant_article_ids=['a']) for q in ['first','third','miss']]
        report = evaluate(fake, cases)
        self.assertAlmostEqual(report['hit_at_1'], 1/3)
        self.assertAlmostEqual(report['hit_at_3'], 2/3)
        self.assertAlmostEqual(report['mrr_at_3'], 4/9)
        with self.assertRaises(ValueError):
            evaluate(fake, [])

    def test_labels_reference_real_articles(self):
        article_ids = {a.id for a in validate_dataset().knowledge_base}
        cases = load_cases()
        self.assertEqual(len({c.id for c in cases}), len(cases))
        self.assertTrue(all(set(c.relevant_article_ids) <= article_ids for c in cases))

    def test_missing_key_and_sdk_response_order(self):
        with self.assertRaisesRegex(ValueError, 'SECURESUPPORT_OPENAI_API_KEY'):
            OpenAIEmbedder(Settings(_env_file=None, openai_api_key=None))
        with patch('app.retrieval.embeddings.OpenAI') as sdk:
            provider = OpenAIEmbedder(Settings(_env_file=None, openai_api_key='test-only-placeholder'))
            sdk.return_value.embeddings.create.return_value = SimpleNamespace(data=[
                SimpleNamespace(index=1, embedding=[0.,1.]), SimpleNamespace(index=0, embedding=[1.,0.])])
            self.assertEqual(provider.embed(['refund','recovery']), [[1.,0.],[0.,1.]])
            sdk.return_value.embeddings.create.assert_called_once_with(
                model='text-embedding-3-small', input=['refund','recovery'], encoding_format='float')


if __name__ == '__main__':
    unittest.main()
