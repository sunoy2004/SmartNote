import { useState, useEffect } from "react";

export interface Note {
  id: string;
  transcript: string;
  summary: string;
  timestamp: string;
}

export const useNotes = () => {
  const [notes, setNotes] = useState<Note[]>([]);

  // Load notes from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem("ai-notes");
    if (stored) {
      setNotes(JSON.parse(stored));
    }
  }, []);

  // Save notes to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem("ai-notes", JSON.stringify(notes));
  }, [notes]);

  const addNote = (transcript: string, summary: string) => {
    const newNote: Note = {
      id: Date.now().toString(),
      transcript,
      summary,
      timestamp: new Date().toISOString(),
    };
    setNotes((prev) => [newNote, ...prev]);
  };

  const deleteNote = (id: string) => {
    setNotes((prev) => prev.filter((note) => note.id !== id));
  };

  return { notes, addNote, deleteNote };
};
