import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { SessionSummary, SessionDetail } from '../services/api';
import { ExplanationChart } from '../components/ExplanationChart';
import { Trash2, ChevronDown, ChevronUp, Clock, AlertCircle, Loader } from 'lucide-react';

const RECOMMENDATION_STYLE: Record<string, string> = {
    STRONG_HIRE:     'bg-green-100 text-green-800',
    RECOMMENDED:     'bg-blue-100 text-blue-800',
    CONSIDER:        'bg-amber-100 text-amber-800',
    NOT_RECOMMENDED: 'bg-red-100 text-red-800',
};

const RECOMMENDATION_LABEL: Record<string, string> = {
    STRONG_HIRE:     'Strong Hire',
    RECOMMENDED:     'Recommended',
    CONSIDER:        'Consider',
    NOT_RECOMMENDED: 'Not Recommended',
};

export const HistoryPage: React.FC = () => {
    const [sessions, setSessions] = useState<SessionSummary[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [expanded, setExpanded] = useState<string | null>(null);
    const [detail, setDetail] = useState<SessionDetail | null>(null);
    const [detailLoading, setDetailLoading] = useState(false);
    const [deleting, setDeleting] = useState<string | null>(null);

    useEffect(() => { loadSessions(); }, []);

    const loadSessions = async () => {
        try {
            const data = await api.listSessions();
            setSessions(data);
        } catch {
            setError('Failed to load sessions. Make sure the backend is running.');
        } finally {
            setLoading(false);
        }
    };

    const handleExpand = async (id: string) => {
        if (expanded === id) {
            setExpanded(null);
            setDetail(null);
            return;
        }
        setExpanded(id);
        setDetail(null);
        setDetailLoading(true);
        try {
            const d = await api.getSession(id);
            setDetail(d);
        } catch {
            setError('Failed to load session details.');
        } finally {
            setDetailLoading(false);
        }
    };

    const handleDelete = async (id: string) => {
        if (!window.confirm('Delete this interview session? This cannot be undone.')) return;
        setDeleting(id);
        try {
            await api.deleteSession(id);
            setSessions(prev => prev.filter(s => s.id !== id));
            if (expanded === id) { setExpanded(null); setDetail(null); }
        } catch {
            setError('Failed to delete session.');
        } finally {
            setDeleting(null);
        }
    };

    if (loading) return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
            <Loader className="w-12 h-12 text-primary-600 animate-spin" />
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-50 py-12">
            <div className="container mx-auto px-4 max-w-4xl">
                <div className="text-center mb-10">
                    <h1 className="text-4xl font-bold text-gray-900 mb-2">Interview History</h1>
                    <p className="text-gray-500">
                        {sessions.length} saved session{sessions.length !== 1 ? 's' : ''}
                    </p>
                </div>

                {error && (
                    <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3">
                        <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
                        <span className="text-red-700">{error}</span>
                    </div>
                )}

                {sessions.length === 0 && !error && (
                    <div className="text-center py-20 text-gray-400">
                        <p className="text-xl">No interviews saved yet.</p>
                        <p className="text-sm mt-2">Complete an interview on the Live Analysis page to see it here.</p>
                    </div>
                )}

                <div className="space-y-4">
                    {sessions.map(session => (
                        <div key={session.id} className="bg-white rounded-2xl shadow-sm border border-gray-100">
                            {/* Session Header */}
                            <div className="p-5 flex items-center justify-between gap-4">
                                <div className="flex items-center gap-4 flex-1 min-w-0">
                                    <span className={`px-3 py-1 rounded-full text-sm font-semibold whitespace-nowrap ${RECOMMENDATION_STYLE[session.recommendation] ?? 'bg-gray-100 text-gray-700'}`}>
                                        {RECOMMENDATION_LABEL[session.recommendation] ?? session.recommendation}
                                    </span>
                                    <div className="flex gap-4 text-sm text-gray-600">
                                        <span>Curiosity: <strong>{session.curiosity_avg?.toFixed(1)}</strong></span>
                                        <span>Critical Thinking: <strong>{session.critical_thinking_avg?.toFixed(1)}</strong></span>
                                        <span>Creativity: <strong>{session.creativity_avg?.toFixed(1)}</strong></span>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3 shrink-0">
                                    <span className="flex items-center gap-1 text-xs text-gray-400">
                                        <Clock className="w-3 h-3" />
                                        {new Date(session.created_at).toLocaleString()}
                                    </span>
                                    <button
                                        onClick={() => handleExpand(session.id)}
                                        className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 transition-colors"
                                    >
                                        {expanded === session.id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                    </button>
                                    <button
                                        onClick={() => handleDelete(session.id)}
                                        disabled={deleting === session.id}
                                        className="p-2 rounded-lg hover:bg-red-50 text-red-400 hover:text-red-600 transition-colors disabled:opacity-40"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>

                            {/* Expanded Detail */}
                            {expanded === session.id && (
                                <div className="border-t border-gray-100 px-5 py-4">
                                    {detailLoading && (
                                        <div className="flex justify-center py-6">
                                            <Loader className="w-6 h-6 text-primary-500 animate-spin" />
                                        </div>
                                    )}
                                    {detail && detail.responses.map((r, i) => (
                                        <div key={r.id} className="mb-6 last:mb-0">
                                            <p className="text-xs font-semibold text-gray-400 uppercase mb-1">Question {i + 1}</p>
                                            <p className="text-gray-700 font-medium mb-2">{r.question_text}</p>
                                            <p className="text-sm text-gray-500 italic mb-3">"{r.transcript?.substring(0, 200)}..."</p>
                                            <div className="flex gap-6 text-sm mb-4">
                                                <span className="text-blue-600 font-semibold">Curiosity: {r.curiosity_score?.toFixed(1)}</span>
                                                <span className="text-purple-600 font-semibold">Critical Thinking: {r.critical_thinking_score?.toFixed(1)}</span>
                                                <span className="text-green-600 font-semibold">Creativity: {r.creativity_score?.toFixed(1)}</span>
                                                <span className="text-gray-500">{r.word_count} words</span>
                                            </div>
                                            {r.explanations && Object.keys(r.explanations).length > 0 && (
                                                <div>
                                                    <p className="text-xs font-semibold text-gray-400 uppercase mb-2">
                                                        SHAP Feature Explanations
                                                    </p>
                                                    <div className="grid md:grid-cols-3 gap-3">
                                                        {(['curiosity', 'critical_thinking', 'creativity'] as const).map(trait =>
                                                            r.explanations[trait]?.length > 0 && (
                                                                <ExplanationChart
                                                                    key={trait}
                                                                    trait={trait}
                                                                    entries={r.explanations[trait]}
                                                                />
                                                            )
                                                        )}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
