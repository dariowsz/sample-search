//
//  ContentView.swift
//  ui-sample-search
//
//  Created by Flor on 21/11/23.
//

/*SwiftUI es un framework de desarrollo de aplicaciones de Apple que permite la creación de aplicaciones para todas las plataformas de Apple (iOS, macOS, watchOS, tvOS) con el mismo código base.

 Permite el desarrollo multiplataforma, Proporciona una apariencia y sensación nativas Y tiene una sintaxis más simple y es más fácil de aprender que AppKit (framework de desarrollo de aplicaciones de escritorio SOLO para macOS).
*/
import SwiftUI

/*ContentView hereda de View. En SwiftUI, una View es cualquier cosa que se pueda dibujar en la pantalla.*/
struct ContentView: View {
    /* Defino dos variables de estado. En SwiftUI, @State es una propiedad que le permite a la vista observar cambios en los valores y actualizar la interfaz de usuario cuando cambian.*/
    @State private var path: String = ""
    @State private var query: String = ""

    /*Cada View en SwiftUI debe tener una propiedad body que describa su contenido y comportamiento.*/
    var body: some View {
        /*VStack es una vista que organiza sus subvistas en una pila vertical.*/
        VStack {
            Text("Sample Search")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.bottom, 20)
                .foregroundColor(.white)
            Button(action: {
                /*NSOpenPanel es una vista que permite al usuario seleccionar un archivo o directorio.*/
                let dialog = NSOpenPanel()
                dialog.canChooseFiles = false
                dialog.canChooseDirectories = true
                dialog.allowsMultipleSelection = false

                /*runModal() muestra el panel y bloquea el resto de la interfaz de usuario hasta que el usuario cierra el panel. Si el usuario selecciona "OK", se ejecuta este bloque:*/
                if dialog.runModal() == NSApplication.ModalResponse.OK {
                    let result = dialog.url
                    if let path = result?.path {
                        self.path = path
                        // Script embeddings y subirlos a la bdd
                    }
                }
            }) {
                Text("Browse")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.green)
                    .cornerRadius(10)
            }
            .buttonStyle(PlainButtonStyle())

            TextField("Path", text: $path)
                .disabled(true)
                .padding()
                .background(Color.black.opacity(0.1))
                .cornerRadius(10)
                .padding(.top, 10)
                .textFieldStyle(PlainTextFieldStyle())
                .foregroundColor(Color.gray)

            TextField("Query", text: $query)
                .padding()
                .background(Color.black.opacity(0.1))
                .cornerRadius(10)
                .padding(.top, 10)
                .textFieldStyle(PlainTextFieldStyle())
                .foregroundColor(Color.gray)

            Button(action: {
                // Script semantic search
            }) {
                /*HStack es una vista que organiza sus subvistas en una pila horizontal.*/
                HStack {
                    Image(systemName: "magnifyingglass")
                    Text("Search")
                }
                .font(.headline)
                .foregroundColor(.white)
                .padding()
                .background(Color.blue)
                .cornerRadius(10)
            }
            .buttonStyle(PlainButtonStyle())
        }
        .padding()
        .background(Color.black)
    }
}
