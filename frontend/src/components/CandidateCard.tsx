import React from 'react';
import type { Candidate } from '../services/api';
import { TrendingUp } from 'lucide-react';

interface CandidateCardProps {
    candidate: Candidate;
    onClick: () => void;
}

export const CandidateCard: React.FC<CandidateCardProps> = ({ candidate, onClick }) => {
    const avgScore = (
        candidate.scores.curiosity +
        candidate.scores.critical_thinking +
        candidate.scores.creativity
    ) / 3;

    const getRecommendationBadge = () => {
        if (avgScore >= 4.0) return { text: 'Strong Hire', color: 'bg-green-100 text-green-800' };
        if (avgScore >= 3.5) return { text: 'Recommended', color: 'bg-blue-100 text-blue-800' };
        if (avgScore >= 3.0) return { text: 'Consider', color: 'bg-amber-100 text-amber-800' };
        return { text: 'Not Recommended', color: 'bg-red-100 text-red-800' };
    };

    const badge = getRecommendationBadge();

    return (
        <div
            onClick={onClick}
            className="card hover:shadow-md transition-shadow cursor-pointer group"
        >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
                <div>
                    <h3 className="font-semibold text-lg text-gray-900 group-hover:text-primary-600 transition-colors">
                        Candidate
                    </h3>
                    <p className="text-sm text-gray-500">{candidate.word_count} words</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${badge.color}`}>
                    {badge.text}
                </span>
            </div>

            {/* Transcript Preview */}
            <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                {candidate.transcript}
            </p>

            {/* Scores */}
            <div className="grid grid-cols-3 gap-3">
                <ScorePill label="Curiosity" score={candidate.scores.curiosity} />
                <ScorePill label="Critical" score={candidate.scores.critical_thinking} />
                <ScorePill label="Creativity" score={candidate.scores.creativity} />
            </div>

            {/* Average */}
            <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Average Score</span>
                <div className="flex items-center">
                    <span className="text-lg font-bold text-gray-900">{avgScore.toFixed(1)}</span>
                    <span className="text-sm text-gray-500 ml-1">/ 5</span>
                </div>
            </div>
        </div>
    );
};

interface ScorePillProps {
    label: string;
    score: number;
}

const ScorePill: React.FC<ScorePillProps> = ({ label, score }) => {
    const getColor = () => {
        if (score >= 4.0) return 'text-green-600 bg-green-50';
        if (score >= 3.5) return 'text-blue-600 bg-blue-50';
        if (score >= 3.0) return 'text-amber-600 bg-amber-50';
        return 'text-red-600 bg-red-50';
    };

    return (
        <div className={`${getColor()} rounded-lg p-2 text-center`}>
            <div className="text-xs font-medium mb-1">{label}</div>
            <div className="text-lg font-bold">{score.toFixed(1)}</div>
        </div>
    );
};
