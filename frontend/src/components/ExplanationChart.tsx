import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { ShapEntry } from '../services/api';

interface ExplanationChartProps {
    trait: string;
    entries: ShapEntry[];
}

export const ExplanationChart: React.FC<ExplanationChartProps> = ({ trait, entries }) => {
    const data = entries.map(e => ({
        feature: e.feature,
        impact: e.impact,
        direction: e.direction,
    }));

    return (
        <div className="bg-gray-50 rounded-xl p-4">
            <p className="text-sm font-semibold text-gray-700 mb-3 capitalize">
                {trait.replace('_', ' ')} — Top Contributing Features
            </p>
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
                        formatter={(value: number, _name: string, props: any) => [
                            `Impact: ${value.toFixed(4)}`,
                            props.payload.direction === 'positive' ? 'Increases score' : 'Decreases score'
                        ]}
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
