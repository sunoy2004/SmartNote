import { useState } from "react";
import { NotesHeader } from "@/components/NotesHeader";
import { VoiceRecorder } from "@/components/VoiceRecorder";
import { NotesList } from "@/components/NotesList";
import { useNotes } from "@/hooks/useNotes";

const Index = () => {
  const { notes, addNote, deleteNote } = useNotes();
  const [isRecording, setIsRecording] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <NotesHeader notesCount={notes.length} />
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <VoiceRecorder 
          onTranscript={addNote}
          isRecording={isRecording}
          setIsRecording={setIsRecording}
        />
        <NotesList notes={notes} onDelete={deleteNote} />
      </main>
    </div>
  );
};

export default Index;
