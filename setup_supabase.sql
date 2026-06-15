-- ==============================================
-- Supabase Setup for Face Recognition Attendance
-- ==============================================
-- Run this in Supabase Dashboard → SQL Editor
-- ==============================================

-- Create attendance table
CREATE TABLE IF NOT EXISTS attendance (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    person_name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    confidence REAL NOT NULL,
    details JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_person_date ON attendance(person_name, date);

-- Enable Row Level Security (recommended by Supabase)
ALTER TABLE attendance ENABLE ROW LEVEL SECURITY;

-- Allow all operations via service_role key (server-side only)
CREATE POLICY "Allow all for service role" ON attendance
    FOR ALL
    USING (true)
    WITH CHECK (true);
