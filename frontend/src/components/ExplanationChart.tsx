import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ShapEntry } from '../services/api';

interface ExplanationChartProps {
    trait: string;
    entries: ShapEntry[];
}

function buildSummary(trait: string, entries: ShapEntry[]): string {
    const traitLabel = trait.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

    if (!entries || entries.length === 0) return '';

    const top = entries[0];
    const second = entries[1];

    const positives = entries.filter(e => e.direction === 'positive');
    const negatives = entries.filter(e => e.direction === 'negative');

    // Both top two go the same direction
    if (second && top.direction === second.direction) {
        const verb = top.direction === 'positive' ? 'driven primarily by' : 'held back primarily by';
        return `Your ${traitLabel} score was ${verb} ${top.feature} and ${second.feature}.`;
    }

    // Top positive, second negative
    if (top.direction === 'positive' && negatives.length > 0) {
        return `Your ${traitLabel} score was boosted by ${top.feature} but reduced by ${negatives[0].feature}.`;
    }

    // Top negative, second positive
    if (top.direction === 'negative' && positives.length > 0) {
        return `Your ${traitLabel} score was held back by ${top.feature} despite ${positives[0].feature} contributing positively.`;
    }

    // Fallback — only one feature
    const verb = top.direction === 'positive' ? 'driven primarily by' : 'held back primarily by';
    return `Your ${traitLabel} score was ${verb} ${top.feature}.`;
}

export const ExplanationChart: React.FC<ExplanationChartProps> = ({ trait, entries }) => {
    const data = entries.map(e => ({
        feature: e.feature,
        impact: e.impact,
        direction: e.direction,
    }));

    const summary = buildSummary(trait, entries);

    return (
        <div className="bg-gray-50 rounded-xl p-4">
            <p className="text-sm font-semibold text-gray-700 mb-1 capitalize">
                {trait.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())} — Top Contributing Features
            </p>
            {summary && (
                <p className="text-xs text-gray-500 italic mb-3">{summary}</p>
            )}
            <ResponsiveContainer width="100%" height={180}>
                <BarChart
                    data={data}
                    layout="vertical"
                    margin={{ top: 0, right: 20, left: 10, bottom: 0 }}
                >
                    <XAxis type="number" domain={[0, 'auto']} tick={{ fontSize: 11 }} />
                    <YAxis
                        type="category"
                        dataKey="feature"
                        width={140}
                        tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                        formatter={(value: number | undefined, _name: string | undefined, props: any) => [
                            `Impact: ${(value ?? 0).toFixed(4)}`,
                            props.payload.direction === 'positive' ? 'Increases score' : 'Decreases score'
                        ] as [string, string]}
                    />
                    <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                        {data.map((entry, index) => (
                            <Cell
                                key={index}
                                fill={entry.direction === 'positive' ? '#4ade80' : '#f87171'}
                            />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-2 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded-sm bg-green-400 inline-block" /> Increases score
                </span>
                <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded-sm bg-red-400 inline-block" /> Decreases score
                </span>
            </div>
        </div>
    );
};
