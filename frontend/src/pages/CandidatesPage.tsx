import React, { useEffect, useState } from 'react';
import { CandidateCard } from '../components/CandidateCard';
import { api } from '../services/api';
import type { Candidate } from '../services/api';
import { Loader, AlertCircle } from 'lucide-react';

export const CandidatesPage: React.FC = () => {
    const [candidates, setCandidates] = useState<Candidate[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);

    useEffect(() => {
        loadCandidates();
    }, []);

    const loadCandidates = async () => {
        try {
            const data = await api.getCandidates();
            setCandidates(data);
        } catch (err) {
            setError('Failed to load candidates. Make sure the backend is running.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleCandidateClick = async (id: string) => {
        try {
            const candidate = await api.getCandidate(id);
            setSelectedCandidate(candidate);
        } catch (err) {
            console.error('Failed to load candidate details:', err);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <Loader className="w-12 h-12 text-primary-600 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="card max-w-md">
                    <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
                    <p className="text-center text-gray-700">{error}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 py-12">
            <div className="container mx-auto px-4">
                <div className="text-center mb-12">
                    <h1 className="text-4xl font-bold mb-4 text-gray-900">Demo Candidates</h1>
                    <p className="text-xl text-gray-600">
                        Explore {candidates.length} pre-assessed candidates
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {candidates.map((candidate) => (
                        <CandidateCard
                            key={candidate.id}
                            candidate={candidate}
                            onClick={() => handleCandidateClick(candidate.id)}
                        />
                    ))}
                </div>
            </div>

            {selectedCandidate && (
                <CandidateModal
                    candidate={selectedCandidate}
                    onClose={() => setSelectedCandidate(null)}
                />
            )}
        </div>
    );
};

interface CandidateModalProps {
    candidate: Candidate;
    onClose: () => void;
}

const CandidateModal: React.FC<CandidateModalProps> = ({ candidate, onClose }) => {
    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={onClose}>
            <div className="bg-white rounded-lg max-w-3xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
                <div className="p-6">
                    <div className="flex justify-between items-start mb-6">
                        <h2 className="text-2xl font-bold text-gray-900">Candidate Details</h2>
                        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">
                            ×
                        </button>
                    </div>

                    <div className="mb-6">
                        <h3 className="font-semibold mb-2 text-gray-900">Full Transcript</h3>
                        <p className="text-gray-700 leading-relaxed">{candidate.transcript}</p>
                        <p className="text-sm text-gray-500 mt-2">{candidate.word_count} words</p>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-6">
                        <ScoreDisplay label="Curiosity" score={candidate.scores.curiosity} />
                        <ScoreDisplay label="Critical Thinking" score={candidate.scores.critical_thinking} />
                        <ScoreDisplay label="Creativity" score={candidate.scores.creativity} />
                    </div>

                    {candidate.reasoning && (
                        <div className="space-y-4">
                            <h3 className="font-semibold text-gray-900">Reasoning</h3>
                            <ReasoningCard title="Curiosity" reasoning={candidate.reasoning.curiosity} />
                            <ReasoningCard title="Critical Thinking" reasoning={candidate.reasoning.critical_thinking} />
                            <ReasoningCard title="Creativity" reasoning={candidate.reasoning.creativity} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

interface ScoreDisplayProps {
    label: string;
    score: number;
}

const ScoreDisplay: React.FC<ScoreDisplayProps> = ({ label, score }) => (
    <div className="card text-center">
        <div className="text-3xl font-bold text-primary-600 mb-1">{score.toFixed(1)}</div>
        <div className="text-sm text-gray-600">{label}</div>
    </div>
);

interface ReasoningCardProps {
    title: string;
    reasoning: string;
}

const ReasoningCard: React.FC<ReasoningCardProps> = ({ title, reasoning }) => (
    <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-2">{title}</h4>
        <p className="text-gray-700 text-sm">{reasoning}</p>
    </div>
);
