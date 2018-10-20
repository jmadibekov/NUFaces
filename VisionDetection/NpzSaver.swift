
import Foundation

extension Npz {
    public func save(to url: URL) throws {
        let data = self.format()
        try data.write(to: url)
    }
    
    public func format() -> Data {
        var entries = [String: Data]()
        for (k, v) in dict {
            entries[k] = v.format()
        }
        return createZip(entries: entries)
    }

}
