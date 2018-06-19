# Code for MIDI analysis of Debussy's "Clair De Lune"
# by Marc Jones
from music21 import *
from collections import Counter as ctr

output_text = ''

# musicxml file created with Finale from the midi file of the same name
filename = 'debussy_clairdelune.musicxml'

# parse the midi file
cdl = converter.parse(filename,format='musicxml')

# create a version of the score simplifying the complexity into succession of chords
cdlchord = cdl.chordify()

# show the midi score as musical notation using musicxml format
cdl.show() # requires external installation of a musicxml viewer such as MuseScore or Finale

# identify the key of the piece
key = cdl.analyze('key')
# print(filename,'is in the key of:',key)
output_text += filename + ' is in the key of: ' + str(key) + '\n'

# identify common chords in the piece
all_chords = []
for chrd in cdlchord.recurse().getElementsByClass('Chord'):
	all_chords.append(chrd.commonName)
	# print(chrd.measureNumber, chrd.beatStr, chrd.commonName, str(chrd))
# print('\nMost Common Chords:\n',ctr(all_chords).most_common(11))
output_text += '\nMost Common Chords:\n' + str(ctr(all_chords).most_common(11))

# identify the key for each measure
measure_keys = []
for m in cdlchord.getElementsByClass('Measure'):
    key = m.analyze('key')
    # print(key)
    measure_keys.append(key)
# print('\nKey of Measures Counted:\n',ctr(measure_keys))
output_text += '\nKey of Measures Counted:\n' + str(ctr(measure_keys)) + '\n'

# track individual key modulation across measures
modulations = []
print(len(measure_keys))
for i,key in enumerate(measure_keys):
	output = str(measure_keys[i]) + ' to ' + str(measure_keys[i+1])
	modulations.append(output)
	if i == len(measure_keys)-2:
		break # the last measure only has one chord which we will analyze in a later section
# print('Modulations Tracked:\n',ctr(modulations).most_common(10))
output_text += '\nModulations Tracked:\n' + str(ctr(modulations).most_common(10)) + '\n'

# identify the last note in the piece
lastNote = cdl.recurse().getElementsByClass('Note')[-1]
# print('\nThe last note played:',str(lastNote))
# print(lastNote.getContextByClass('KeySignature'))
output_text += '\nThe last note played:' + str(lastNote) + '\n'
output_text += str(lastNote.getContextByClass('KeySignature'))

outfile = open('cdl_midi_anal.txt','w')
outfile.write(output_text)
outfile.close()